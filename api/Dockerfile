FROM python:3.9-slim

WORKDIR /app

# 安裝依賴
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼和模型
COPY api/ ./api/
COPY models/ ./models/
COPY src/ ./src/

# 暴露端口
EXPOSE 8000

# 啟動服務
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]