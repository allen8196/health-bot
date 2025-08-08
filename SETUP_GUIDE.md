# 健康陪伴機器人 設置指南

## 快速啟動

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 配置環境變數

1. 複製配置模板：
```bash
copy config_template.env .env
```

2. 編輯 `.env` 文件，填入您的 OpenAI API 密鑰：
```
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
MODEL_NAME=gpt-4o-mini
SIMILARITY_THRESHOLD=0.7
```

### 3. 啟動 Milvus 向量資料庫

使用 Docker Compose 啟動 Milvus：
```bash
docker-compose up -d
```

### 4. 載入 COPD 問答資料

```bash
python load_article.py
```

### 5. 運行機器人

```bash
python main.py
```

## 功能說明

- **風險檢查**：自動偵測危險內容（自殺、暴力等）
- **健康陪伴**：以台語風格關懷長者
- **知識搜尋**：從 COPD 資料庫中搜尋相關資訊
- **個管師通報**：緊急情況自動通報
- **對話記錄**：自動保存與摘要對話歷史

## 故障排除

如果遇到錯誤，請檢查：
1. OpenAI API 密鑰是否正確設置
2. Milvus 是否正常運行
3. 所有依賴是否正確安裝