from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import json
import os
import logging
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field
import uuid
import time
from datetime import datetime
import asyncio

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


app = FastAPI()

BACKEND_TYPE = os.getenv("BACKEND_TYPE", "openai").lower()
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8856")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8080"))

def safe_json_loads(data: Union[bytes, str]) -> Dict[str, Any]:
    """
    安全的JSON解析函数，自动处理bytes/string类型转换和详细错误诊断
    """
    try:
        # 检测数据类型并进行适当转换
        if isinstance(data, bytes):
            # 尝试UTF-8解码
            try:
                data_str = data.decode('utf-8')
            except UnicodeDecodeError as e:
                logger.error(f"UTF-8解码失败: {e}")
                raise json.JSONDecodeError(f"请求体编码错误: {e}", str(data[:50]), 0)
        elif isinstance(data, str):
            data_str = data
        else:
            logger.error(f"不支持的数据类型: {type(data)}")
            raise json.JSONDecodeError(f"不支持的数据类型: {type(data)}", str(data), 0)
        
        # 验证数据不为空
        if not data_str.strip():
            logger.error("请求体为空")
            raise json.JSONDecodeError("请求体为空", data_str, 0)
        
        # 尝试JSON解析
        return json.loads(data_str)
        
    except json.JSONDecodeError as e:
        # 记录详细的错误诊断信息
        logger.error(f"JSON解析错误 - 位置: 第{e.lineno}行第{e.colno}列 (字符{e.pos})")
        logger.error(f"错误消息: {e.msg}")
        if isinstance(data, (bytes, str)):
            # 安全地显示前50个字符用于调试，避免敏感信息泄露
            preview = str(data)[:50] if len(str(data)) > 50 else str(data)
            logger.error(f"请求体预览: {preview}...")
            logger.error(f"数据类型: {type(data)}, 长度: {len(data)}")
        raise
    except Exception as e:
        logger.error(f"JSON解析过程中发生未知错误: {e}")
        raise json.JSONDecodeError(f"JSON解析失败: {e}", str(data)[:50] if data else "", 0)

class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

class AnthropicContent(BaseModel):
    type: str
    text: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None

class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[AnthropicContent]]

class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    metadata: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None

def convert_openai_to_anthropic(openai_req: Dict[str, Any]) -> Dict[str, Any]:
    anthropic_messages = []
    system_content = None
    
    for msg in openai_req["messages"]:
        role = msg["role"]
        content = msg.get("content", "")
        
        if role == "system":
            system_content = content
            continue
            
        if role == "function":
            role = "assistant"
            
        anthropic_content = []
        
        if isinstance(content, str):
            if content:
                anthropic_content.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if block["type"] == "text":
                    anthropic_content.append({"type": "text", "text": block["text"]})
                elif block["type"] == "image_url":
                    image_data = block["image_url"]
                    if isinstance(image_data, dict):
                        url = image_data.get("url", "")
                    else:
                        url = image_data
                    
                    if url.startswith("data:"):
                        media_type, data = url.split(",", 1)
                        media_type = media_type.split(";")[0].split(":")[1]
                        anthropic_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data
                            }
                        })
        
        if "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                anthropic_content.append({
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": json.loads(tool_call["function"]["arguments"])
                })
        
        if "tool_call_id" in msg and msg["tool_call_id"]:
            tool_result_content = content if isinstance(content, str) else json.dumps(content)
            anthropic_content.append({
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": tool_result_content
            })
        
        if anthropic_content:
            anthropic_messages.append({
                "role": "user" if role == "user" else "assistant",
                "content": anthropic_content
            })
    
    anthropic_req = {
        "model": openai_req["model"],
        "messages": anthropic_messages,
        "max_tokens": openai_req.get("max_tokens", 4096),
        "temperature": openai_req.get("temperature", 1.0),
        "stream": openai_req.get("stream", False)
    }
    
    if system_content:
        anthropic_req["system"] = system_content
    
    if "stop" in openai_req:
        stop = openai_req["stop"]
        anthropic_req["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    
    if "top_p" in openai_req:
        anthropic_req["top_p"] = openai_req["top_p"]
    
    if "tools" in openai_req and openai_req["tools"]:
        anthropic_tools = []
        for tool in openai_req["tools"]:
            if tool["type"] == "function":
                anthropic_tools.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"]["parameters"]
                })
        anthropic_req["tools"] = anthropic_tools
    
    if "tool_choice" in openai_req:
        choice = openai_req["tool_choice"]
        if choice == "auto":
            anthropic_req["tool_choice"] = {"type": "auto"}
        elif choice == "none":
            anthropic_req["tool_choice"] = {"type": "none"}
        elif isinstance(choice, dict) and choice.get("type") == "function":
            anthropic_req["tool_choice"] = {
                "type": "tool",
                "name": choice["function"]["name"]
            }
    
    return anthropic_req

def convert_anthropic_to_openai(anthropic_req: Dict[str, Any]) -> Dict[str, Any]:
    openai_messages = []
    
    if "system" in anthropic_req and anthropic_req["system"]:
        system = anthropic_req["system"]
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            system_text = ""
            for block in system:
                if block.get("type") == "text":
                    system_text += block.get("text", "") + "\n"
            if system_text:
                openai_messages.append({"role": "system", "content": system_text.strip()})
    
    for msg in anthropic_req["messages"]:
        role = msg["role"]
        content = msg["content"]
        
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            openai_content = []
            tool_calls = []
            tool_results = []
            
            for block in content:
                block_type = block.get("type")
                
                if block_type == "text":
                    openai_content.append({
                        "type": "text",
                        "text": block["text"]
                    })
                elif block_type == "image":
                    source = block["source"]
                    if source["type"] == "base64":
                        url = f"data:{source['media_type']};base64,{source['data']}"
                        openai_content.append({
                            "type": "image_url",
                            "image_url": {"url": url}
                        })
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"])
                        }
                    })
                elif block_type == "tool_result":
                    tool_results.append({
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"]
                    })
            
            if openai_content or tool_calls:
                msg_dict = {"role": role}
                if openai_content:
                    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
                        msg_dict["content"] = openai_content[0]["text"]
                    else:
                        msg_dict["content"] = openai_content
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                openai_messages.append(msg_dict)
            
            for tool_result in tool_results:
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": tool_result["content"]
                })
    
    openai_req = {
        "model": anthropic_req["model"],
        "messages": openai_messages,
        "temperature": anthropic_req.get("temperature", 1.0),
        "stream": anthropic_req.get("stream", False)
    }
    
    if "max_tokens" in anthropic_req:
        openai_req["max_tokens"] = anthropic_req["max_tokens"]
    
    if "stop_sequences" in anthropic_req:
        openai_req["stop"] = anthropic_req["stop_sequences"]
    
    if "top_p" in anthropic_req:
        openai_req["top_p"] = anthropic_req["top_p"]
    
    if "tools" in anthropic_req and anthropic_req["tools"]:
        openai_tools = []
        for tool in anthropic_req["tools"]:
            # 安全获取工具参数，支持多种字段名
            tool_params = None
            if "input_schema" in tool:
                tool_params = tool["input_schema"]
            elif "parameters" in tool:
                tool_params = tool["parameters"]
            elif "schema" in tool:
                tool_params = tool["schema"]
            else:
                # 如果没有找到参数定义，使用空对象并记录警告
                logger.warning(f"Tool '{tool.get('name', 'unknown')}' missing parameter schema, using empty schema")
                tool_params = {"type": "object", "properties": {}}
            
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", "unknown_function"),
                    "description": tool.get("description", ""),
                    "parameters": tool_params
                }
            })
        openai_req["tools"] = openai_tools
    
    if "tool_choice" in anthropic_req:
        choice = anthropic_req["tool_choice"]
        if choice["type"] == "auto":
            openai_req["tool_choice"] = "auto"
        elif choice["type"] == "none":
            openai_req["tool_choice"] = "none"
        elif choice["type"] == "tool":
            openai_req["tool_choice"] = {
                "type": "function",
                "function": {"name": choice["name"]}
            }
    
    return openai_req

def convert_openai_response_to_anthropic(openai_resp: Dict[str, Any]) -> Dict[str, Any]:
    # Check if this is an error response
    if "error" in openai_resp:
        return {
            "type": "error",
            "error": {
                "type": openai_resp["error"].get("type", "api_error"),
                "message": openai_resp["error"].get("message", "Unknown error")
            }
        }
    
    # Check if choices exist
    if "choices" not in openai_resp or not openai_resp["choices"]:
        return {
            "type": "error",
            "error": {
                "type": "invalid_response",
                "message": "No choices in OpenAI response"
            }
        }
    
    choice = openai_resp["choices"][0]
    message = choice.get("message", {})
    
    anthropic_content = []
    
    if "content" in message and message["content"]:
        anthropic_content.append({
            "type": "text",
            "text": message["content"]
        })
    
    if "tool_calls" in message and message["tool_calls"]:
        for tool_call in message["tool_calls"]:
            anthropic_content.append({
                "type": "tool_use",
                "id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "input": json.loads(tool_call["function"]["arguments"])
            })
    
    # Ensure content is not empty
    if not anthropic_content:
        anthropic_content.append({
            "type": "text",
            "text": ""
        })
    
    stop_reason = "end_turn"
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "stop":
        stop_reason = "stop_sequence"
    elif finish_reason == "tool_calls":
        stop_reason = "tool_use"
    
    usage = openai_resp.get("usage", {})
    return {
        "id": f"msg_{openai_resp.get('id', uuid.uuid4().hex)}",
        "type": "message",
        "role": "assistant",
        "content": anthropic_content,
        "model": openai_resp.get("model", "unknown"),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0)
        }
    }

def convert_anthropic_response_to_openai(anthropic_resp: Dict[str, Any]) -> Dict[str, Any]:
    # Check if this is an error response
    if anthropic_resp.get("type") == "error":
        return {
            "error": {
                "message": anthropic_resp.get("error", {}).get("message", "Unknown error"),
                "type": anthropic_resp.get("error", {}).get("type", "api_error"),
                "code": None
            }
        }
    
    # Ensure content exists
    if "content" not in anthropic_resp:
        return {
            "error": {
                "message": "No content in Anthropic response",
                "type": "invalid_response",
                "code": None
            }
        }
    
    content_text = ""
    tool_calls = []
    
    for block in anthropic_resp["content"]:
        if block["type"] == "text":
            content_text += block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block["input"])
                }
            })
    
    message = {"role": "assistant"}
    if content_text:
        message["content"] = content_text
    else:
        message["content"] = None
        
    if tool_calls:
        message["tool_calls"] = tool_calls
    
    finish_reason = "stop"
    stop_reason = anthropic_resp.get("stop_reason")
    if stop_reason == "max_tokens":
        finish_reason = "length"
    elif stop_reason == "stop_sequence":
        finish_reason = "stop"
    elif stop_reason == "tool_use":
        finish_reason = "tool_calls"
    
    usage = anthropic_resp.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": anthropic_resp.get("model", "unknown"),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": None
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        },
        "system_fingerprint": None
    }

async def stream_response_handler(response_generator) -> AsyncGenerator[bytes, None]:
    """Handle streaming response with proper buffering"""
    buffer = ""
    try:
        async for chunk in response_generator:
            buffer += chunk.decode("utf-8")
            
            # Process complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                
                if line:
                    # Yield the complete line with proper formatting
                    if line.startswith("data: "):
                        yield (line + "\n\n").encode("utf-8")
                    elif line.startswith("event: "):
                        yield (line + "\n").encode("utf-8")
                    else:
                        yield (line + "\n").encode("utf-8")
                        
    except Exception as e:
        logger.error(f"Error in stream handler: {str(e)}")
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode("utf-8")

async def stream_openai_to_anthropic(response: httpx.Response) -> AsyncGenerator[str, None]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    
    # Send message_start event
    message_data = {
        'type': 'message_start',
        'message': {
            'id': message_id,
            'type': 'message',
            'role': 'assistant',
            'content': [],
            'model': '',
            'stop_reason': None,
            'stop_sequence': None,
            'usage': {
                'input_tokens': 0,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
                'output_tokens': 0
            }
        }
    }
    yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
    
    # Start with text content block
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    
    # Send a ping to keep the connection alive
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
    
    tool_index = None
    current_tool_call = None
    tool_content = ""
    accumulated_text = ""
    text_sent = False
    text_block_closed = False
    input_tokens = 0
    output_tokens = 0
    has_sent_stop_reason = False
    last_tool_index = 0
    tool_call_map = {}  # Maps OpenAI tool call index to Anthropic content block index
    
    async for line in response.aiter_lines():
        if not line or not line.startswith("data: "):
            continue
            
        data = line[6:]
        if data == "[DONE]":
            break
            
        try:
            chunk = json.loads(data)
            if not chunk.get("choices"):
                continue
                
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            
            # Check for usage data
            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]
                if "prompt_tokens" in usage:
                    input_tokens = usage["prompt_tokens"]
                if "completion_tokens" in usage:
                    output_tokens = usage["completion_tokens"]
            
            # Handle text content
            if "content" in delta and delta["content"]:
                accumulated_text += delta["content"]
                
                # Always emit text deltas if no tool calls started
                if tool_index is None and not text_block_closed:
                    text_sent = True
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"
            
            # Handle tool calls
            if "tool_calls" in delta and delta["tool_calls"]:
                # First tool call we've seen - need to handle text properly
                if tool_index is None:
                    # If we've been streaming text, close that text block
                    if text_sent and not text_block_closed:
                        text_block_closed = True
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    # If we've accumulated text but not sent it, emit it now
                    elif accumulated_text and not text_sent and not text_block_closed:
                        text_sent = True
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                        text_block_closed = True
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                    # Close text block even if we haven't sent anything
                    elif not text_block_closed:
                        text_block_closed = True
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                
                for tc in delta["tool_calls"]:
                    tc_index = tc.get("index", 0)
                    
                    # Check if this is a new tool or a continuation
                    if tc_index not in tool_call_map:
                        # New tool call
                        last_tool_index += 1
                        tool_call_map[tc_index] = last_tool_index
                        anthropic_tool_index = last_tool_index
                        
                        # Extract tool info
                        function = tc.get("function", {})
                        name = function.get("name", "")
                        tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                        
                        # Start a new tool_use block
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                        tool_index = tc_index
                        tool_content = ""
                    else:
                        # Continuation of existing tool call
                        anthropic_tool_index = tool_call_map[tc_index]
                    
                    # Tool call arguments
                    if "function" in tc and "arguments" in tc["function"] and tc["function"]["arguments"]:
                        args_json = tc["function"]["arguments"]
                        tool_content += args_json
                        
                        # Send the update
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
            
            # Handle finish reason
            if choice.get("finish_reason") and not has_sent_stop_reason:
                has_sent_stop_reason = True
                
                # Close any open tool call blocks
                if tool_index is not None:
                    for i in range(1, last_tool_index + 1):
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                
                # If we accumulated text but never sent or closed text block, do it now
                if not text_block_closed:
                    if accumulated_text and not text_sent:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                
                # Map OpenAI finish_reason to Anthropic stop_reason
                finish_reason = choice["finish_reason"]
                stop_reason = "end_turn"
                if finish_reason == "length":
                    stop_reason = "max_tokens"
                elif finish_reason == "tool_calls":
                    stop_reason = "tool_use"
                elif finish_reason == "stop":
                    stop_reason = "end_turn"
                
                # Send message_delta with stop reason and usage
                usage = {"output_tokens": output_tokens}
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse chunk: {data}")
            continue
    
    # If we didn't get a finish reason, close any open blocks
    if not has_sent_stop_reason:
        # Close any open tool call blocks
        if tool_index is not None:
            for i in range(1, last_tool_index + 1):
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
        
        # Close the text content block
        if not text_block_closed:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        
        # Send final message_delta with usage
        usage = {"output_tokens": output_tokens}
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
    
    # Send message_stop event
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    
    # Send final [DONE] marker
    yield "data: [DONE]\n\n"

async def stream_anthropic_to_openai_from_sse(sse_message: str) -> AsyncGenerator[str, None]:
    """Convert a single SSE message from Anthropic to OpenAI format"""
    # This function will be called for each complete SSE message
    # and needs to parse it and convert to OpenAI format
    # For now, just pass it through - we'll implement the conversion logic later
    yield sse_message

async def stream_anthropic_to_openai(response: httpx.Response) -> AsyncGenerator[str, None]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    tool_calls = []
    tool_index_map = {}  # Maps Anthropic block index to tool call index
    first_chunk = True
    
    async for line in response.aiter_lines():
        if not line or not line.startswith("data: "):
            continue
            
        data = line[6:]
        if data == "[DONE]":
            yield "data: [DONE]\n\n"
            break
            
        try:
            event = json.loads(data)
            event_type = event.get("type")
            
            if event_type == "message_start":
                # Send initial role chunk
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": event.get("message", {}).get("model", ""),
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }]
                }
                if first_chunk:
                    first_chunk = False
                yield f"data: {json.dumps(chunk)}\n\n"
            
            elif event_type == "content_block_start":
                block = event["content_block"]
                block_index = event.get("index", 0)
                
                if block["type"] == "tool_use":
                    tool_index = len(tool_calls)
                    tool_index_map[block_index] = tool_index
                    
                    # Send initial tool call chunk
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": tool_index,
                                    "id": block["id"],
                                    "type": "function",
                                    "function": {
                                        "name": block["name"],
                                        "arguments": ""
                                    }
                                }]
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": ""
                        }
                    })
            
            elif event_type == "content_block_delta":
                delta = event["delta"]
                block_index = event.get("index", 0)
                
                if delta["type"] == "text_delta":
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta["text"]},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                elif delta["type"] == "input_json_delta":
                    tool_index = tool_index_map.get(block_index, 0)
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": tool_index,
                                    "function": {
                                        "arguments": delta["partial_json"]
                                    }
                                }]
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            elif event_type == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason")
                finish_reason = "stop"
                if stop_reason == "max_tokens":
                    finish_reason = "length"
                elif stop_reason == "tool_use":
                    finish_reason = "tool_calls"
                
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                    }],
                    "usage": event.get("usage", None)
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse event: {data}")
            continue

async def forward_request_stream(
    path: str,
    method: str,
    headers: Dict[str, str],
    body: Optional[bytes] = None,
    params: Optional[Dict[str, Any]] = None
):
    url = f"{BACKEND_BASE_URL}{path}"
    
    forward_headers = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower in ["authorization", "content-type", "accept", "x-api-key"]:
            forward_headers[key] = value
    
    logger.debug(f"Forwarding streaming request to: {url}")
    logger.debug(f"Headers: {forward_headers}")
    if body:
        logger.debug(f"Body: {body[:500]}...")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        async with client.stream(
            method=method,
            url=url,
            headers=forward_headers,
            content=body,
            params=params
        ) as response:
            logger.debug(f"Response status: {response.status_code}")
            
            if response.status_code >= 400:
                error_text = await response.aread()
                logger.error(f"Backend error response: {error_text}")
                raise HTTPException(status_code=response.status_code, detail=error_text.decode())
            
            # For streaming, we'll yield chunks
            async for chunk in response.aiter_bytes():
                yield chunk

async def forward_request(
    path: str,
    method: str,
    headers: Dict[str, str],
    body: Optional[bytes] = None,
    params: Optional[Dict[str, Any]] = None
):
    url = f"{BACKEND_BASE_URL}{path}"
    
    forward_headers = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower in ["authorization", "content-type", "accept", "x-api-key"]:
            forward_headers[key] = value
    
    logger.debug(f"Forwarding request to: {url}")
    logger.debug(f"Headers: {forward_headers}")
    if body:
        logger.debug(f"Body: {body[:500]}...")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        response = await client.request(
            method=method,
            url=url,
            headers=forward_headers,
            content=body,
            params=params
        )
        
        logger.debug(f"Response status: {response.status_code}")
        
        if response.status_code >= 400:
            error_text = response.text
            logger.error(f"Backend error response: {error_text}")
            
        return response

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    body = await request.body()
    headers = dict(request.headers)
    
    try:
        openai_req = safe_json_loads(body)
        
        if BACKEND_TYPE == "anthropic":
            anthropic_req = convert_openai_to_anthropic(openai_req)
            
            if "authorization" in headers:
                headers["x-api-key"] = headers["authorization"].replace("Bearer ", "")
                del headers["authorization"]
            
            if openai_req.get("stream"):
                # For streaming, we need to use httpx.stream properly
                async def stream_generator():
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                        async with client.stream(
                            "POST",
                            f"{BACKEND_BASE_URL}/v1/messages",
                            headers={k: v for k, v in headers.items() 
                                   if k.lower() in ["authorization", "content-type", "accept", "x-api-key"]},
                            content=json.dumps(anthropic_req).encode()
                        ) as response:
                            if response.status_code >= 400:
                                error_text = await response.aread()
                                logger.error(f"Backend error response: {error_text}")
                                yield f"data: {{\"error\": \"{error_text.decode()}\"}}\n\n"
                                return
                            
                            # Pass the response object to the converter
                            async for chunk in stream_anthropic_to_openai(response):
                                yield chunk
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )
            else:
                response = await forward_request(
                    "/v1/messages",
                    "POST",
                    headers,
                    json.dumps(anthropic_req).encode()
                )
                
                response_text = response.text
                logger.debug(f"Response from backend: {response_text[:500]}...")
                
                try:
                    anthropic_resp = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse response as JSON: {e}")
                    logger.error(f"Response text: {response_text}")
                    raise
                
                openai_resp = convert_anthropic_response_to_openai(anthropic_resp)
                return JSONResponse(content=openai_resp)
        else:
            if openai_req.get("stream"):
                # For streaming with OpenAI backend, properly handle the stream
                async def stream_generator():
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                        async with client.stream(
                            request.method,
                            f"{BACKEND_BASE_URL}{request.url.path}",
                            headers={k: v for k, v in headers.items() 
                                   if k.lower() in ["authorization", "content-type", "accept", "x-api-key"]},
                            content=body
                        ) as response:
                            if response.status_code >= 400:
                                error_text = await response.aread()
                                logger.error(f"Backend error response: {error_text}")
                                yield f"data: {{\"error\": \"{error_text.decode()}\"}}\n\n"
                                return
                            
                            async for chunk in response.aiter_bytes():
                                yield chunk
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )
            else:
                response = await forward_request(
                    request.url.path,
                    request.method,
                    headers,
                    body
                )
                return JSONResponse(content=response.json())
    
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        return JSONResponse(
            content={"error": {"message": str(e), "type": "proxy_error"}},
            status_code=500
        )

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.body()
    headers = dict(request.headers)
    
    try:
        anthropic_req = safe_json_loads(body)
        
        if BACKEND_TYPE == "openai":
            openai_req = convert_anthropic_to_openai(anthropic_req)
            
            if "x-api-key" in headers:
                headers["authorization"] = f"Bearer {headers['x-api-key']}"
                del headers["x-api-key"]
            
            if anthropic_req.get("stream"):
                # For streaming, handle the response directly
                async def stream_generator():
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                        async with client.stream(
                            "POST",
                            f"{BACKEND_BASE_URL}/v1/chat/completions",
                            headers={k: v for k, v in headers.items() 
                                   if k.lower() in ["authorization", "content-type", "accept", "x-api-key"]},
                            content=json.dumps(openai_req).encode()
                        ) as response:
                            if response.status_code >= 400:
                                error_text = await response.aread()
                                logger.error(f"Backend error response: {error_text}")
                                yield f"event: error\ndata: {{\"error\": \"{error_text.decode()}\"}}\n\n"
                                return
                            
                            # Pass the response object to the converter
                            async for chunk in stream_openai_to_anthropic(response):
                                yield chunk
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )
            else:
                response = await forward_request(
                    "/v1/chat/completions",
                    "POST",
                    headers,
                    json.dumps(openai_req).encode()
                )
                
                response_text = response.text
                logger.debug(f"Response from backend: {response_text[:500]}...")
                
                try:
                    openai_resp = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse response as JSON: {e}")
                    logger.error(f"Response text: {response_text}")
                    raise
                
                anthropic_resp = convert_openai_response_to_anthropic(openai_resp)
                return JSONResponse(content=anthropic_resp)
        else:
            if anthropic_req.get("stream"):
                # For streaming with Anthropic backend, properly handle the stream
                async def stream_generator():
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                        async with client.stream(
                            request.method,
                            f"{BACKEND_BASE_URL}{request.url.path}",
                            headers={k: v for k, v in headers.items() 
                                   if k.lower() in ["authorization", "content-type", "accept", "x-api-key"]},
                            content=body
                        ) as response:
                            if response.status_code >= 400:
                                error_text = await response.aread()
                                logger.error(f"Backend error response: {error_text}")
                                yield f"event: error\ndata: {{\"error\": \"{error_text.decode()}\"}}\n\n"
                                return
                            
                            async for chunk in response.aiter_bytes():
                                yield chunk
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )
            else:
                response = await forward_request(
                    request.url.path,
                    request.method,
                    headers,
                    body
                )
                return JSONResponse(content=response.json())
    
    except json.JSONDecodeError as e:
        logger.error(f"Error in messages - JSON解析失败: {str(e)}")
        logger.error(f"请求详情 - Content-Type: {headers.get('content-type', 'unknown')}")
        logger.error(f"请求详情 - Content-Length: {headers.get('content-length', 'unknown')}")
        return JSONResponse(
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"无效的JSON格式: {str(e)}"
                }
            },
            status_code=400
        )
    except Exception as e:
        logger.error(f"Error in messages: {str(e)}")
        return JSONResponse(
            content={
                "type": "error",
                "error": {
                    "type": "proxy_error",
                    "message": str(e)
                }
            },
            status_code=500
        )

@app.get("/v1/models")
async def list_models(request: Request):
    headers = dict(request.headers)
    
    response = await forward_request(
            request.url.path,
            request.method,
            headers
        )
    return JSONResponse(content=response.json())

@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body = await request.body()
    headers = dict(request.headers)
    
    if BACKEND_TYPE == "openai":
        return JSONResponse(
            content={
                "type": "error",
                "error": {
                    "type": "not_supported_error",
                    "message": "Token counting endpoint is not supported by OpenAI backend"
                }
            },
            status_code=400
        )
    else:
        response = await forward_request(
            request.url.path,
            request.method,
            headers,
            body
        )
        return JSONResponse(content=response.json())

@app.get("/")
async def health_check():
    return {"status": "healthy", "backend_type": BACKEND_TYPE, "backend_url": BACKEND_BASE_URL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)