# Format Proxy

一个支持 OpenAI 和 Anthropic API 格式相互转换的透明代理服务。

## 功能特性

- **双向格式转换**：支持 OpenAI ↔ Anthropic 格式的自动转换
- **透明传输**：客户端 API 密钥透明传输到后端
- **完整功能支持**：
  - 文本对话
  - 图片输入
  - 工具调用（Function Calling）
  - 流式响应
- **多端点支持**：
  - OpenAI: `/v1/chat/completions`, `/v1/models`
  - Anthropic: `/v1/messages`, `/v1/messages/count_tokens`

## 快速开始

### 使用 Docker Compose

1. 克隆项目并创建环境配置：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件配置后端：
```env
BACKEND_TYPE=openai  # 后端类型：openai 或 anthropic
BACKEND_BASE_URL=https://api.openai.com  # 后端 API 地址
PROXY_PORT=8080
```

3. 启动服务：
```bash
docker-compose up -d
```

### 本地运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 设置环境变量并运行：
```bash
export BACKEND_TYPE=openai
export BACKEND_BASE_URL=https://api.openai.com
python format_proxy.py
```

## 使用示例

### OpenAI 客户端访问 Anthropic 后端

配置：
```env
BACKEND_TYPE=anthropic
BACKEND_BASE_URL=https://api.anthropic.com
```

客户端代码：
```python
import openai

openai.api_base = "http://localhost:8080/v1"
openai.api_key = "your-anthropic-api-key"

response = openai.ChatCompletion.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Anthropic 客户端访问 OpenAI 后端

配置：
```env
BACKEND_TYPE=openai
BACKEND_BASE_URL=https://api.openai.com
```

客户端代码：
```python
import anthropic

client = anthropic.Anthropic(
    api_key="your-openai-api-key",
    base_url="http://localhost:8080"
)

response = client.messages.create(
    model="gpt-4",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| BACKEND_TYPE | 后端 API 类型 (openai/anthropic) | openai |
| BACKEND_BASE_URL | 后端 API 基础 URL | https://api.openai.com |
| PROXY_PORT | 代理服务监听端口 | 8080 |
| LOG_LEVEL | 日志级别 (DEBUG/INFO/WARNING/ERROR) | INFO |

## 架构设计

- **格式检测**：根据请求端点路径自动识别 API 格式
- **透明代理**：保持客户端认证信息，直接传递到后端
- **错误处理**：自动转换错误响应格式，保持客户端兼容性
- **流式处理**：支持 SSE (Server-Sent Events) 流式响应转换