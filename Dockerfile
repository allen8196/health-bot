# 使用官方的 Python 3.11 映像作為基礎
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 將依賴管理檔案複製到容器中
COPY requirements.txt .

# 安裝所有 Python 依賴套件
# --no-cache-dir 選項可以減少映像大小
RUN pip install --no-cache-dir -r requirements.txt

# 將整個專案的程式碼複製到容器的工作目錄中
COPY . .

# 預設執行的腳本於docker-compose.yml中設定