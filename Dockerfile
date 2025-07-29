FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY format_proxy.py .

EXPOSE 8080

CMD ["uvicorn", "format_proxy:app", "--host", "0.0.0.0", "--port", "8080"]