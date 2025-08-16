# 智慧健康陪伴機器人 (Health Companion Bot)

這是一個專為高齡長者設計的智慧健康陪伴專案，旨在透過 LINE Bot 提供個人化的日常關懷、健康資訊查詢，以及基於對話歷史的主動關懷。

專案核心採用了先進的多 Agent 協作架構與三層式記憶體系，使其不僅能回應使用者查詢，更能「記住」過去的互動，並在適當時機主動給予溫暖的問候。

## 系統架構概覽

本系統圍繞著一個 **三層式記憶體系** 和 **多 Agent 協作流程** 進行設計：

1.  **短期記憶 (STM)**: 使用 **Redis** 儲存近期的完整對話，確保對話的即時連貫性。
2.  **長期記憶 (LTM)**: 使用 **Milvus** 儲存由對話精煉出的「最終摘要」，透過向量檢索 (RAG) 實現對過去特定事件的精準回憶。
3.  **使用者畫像 (Profile)**: 使用 **PostgreSQL** 的 `JSONB` 欄位儲存從 LTM 中提煉出的、關於使用者的長期關鍵事實，用於實現深度個人化與主動關懷。

![](https://i.imgur.com/your-architecture-diagram.png) ---

## 快速設置與啟動指南

請遵照以下步驟，從零開始建立完整的開發與測試環境。

### 〇、前置準備

1.  **安裝 Poetry**: 用於管理 Python 虛擬環境與依賴。請參考 [Poetry 官方文件](https://python-poetry.org/docs/#installation)。
2.  **安裝 Docker Desktop**: 用於一鍵啟動所有後端資料庫服務。請從 [Docker 官網](https://www.docker.com/products/docker-desktop/)下載並安裝。
3.  **安裝 ngrok**: 用於建立安全的公網通道，以便 LINE 平台能與您的本機服務溝通。請參考 [ngrok 官網](https://ngrok.com/download)下載並完成 `authtoken` 認證。
4.  **建立 LINE Bot 頻道**: 前往 [LINE Developers Console](https://developers.line.biz/console/) 建立一個 Messaging API 頻道，並取得 `Channel Access Token` 和 `Channel Secret`。

### 一、環境變數設定

1.  編輯環境變數 `.env` 檔案，並填入所有必要的金鑰與設定。這一步至關重要。
    ```ini
    # .env

    # --- OpenAI API Configuration ---
    OPENAI_API_KEY="sk-..." # 請填入OpenAI API Key
    MODEL_NAME=gpt-4o-mini # 可自行切換模型

    # --- LINE Bot Configuration ---
    LINE_CHANNEL_ACCESS_TOKEN="your_line_channel_access_token" # 請填入Line Channel Access Token
    LINE_CHANNEL_SECRET="your_line_channel_secret" # 請填入Line Secret

    # --- PostgreSQL Configuration ---
    POSTGRES_HOST=localhost
    POSTGRES_PORT=5432
    POSTGRES_DB=senior_health
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=your_strong_password # 請設定密碼

    # --- Redis & Milvus Configuration ---
    REDIS_HOST=localhost
    REDIS_PORT=6379
    MILVUS_URI=http://localhost:19530
    MEM_COLLECTION=user_memory

    # ... 其他設定 ...
    ```

### 二、啟動後端服務

1.  **啟動 Docker 容器**:
    在專案根目錄打開終端機，執行以下指令來啟動 PostgreSQL, Redis, 和 Milvus。
    ```bash
    docker compose up -d
    ```

2.  **建立虛擬環境並安裝依賴**:
    ```bash
    # 使用 Poetry 安裝所有依賴並建立虛擬環境
    poetry install

    # 啟動 Poetry 的虛擬環境
    poetry shell
    ```

### 三、資料庫初始化

1.  **初始化 PostgreSQL**:
    * 使用 DBeaver 等 SQL 工具連接到您在 Docker 中啟動的 PostgreSQL。
    * 執行 `database.sql` 檔案中的所有內容，以建立 `senior_users` 等核心表格。
    * 接著，執行以下指令來擴充 Profile 相關欄位：
        ```sql
        CREATE EXTENSION IF NOT EXISTS vector;
        
        ALTER TABLE senior_users ADD COLUMN IF NOT EXISTS profile_personal_background JSONB;
        ALTER TABLE senior_users ADD COLUMN IF NOT EXISTS profile_health_status JSONB;
        ALTER TABLE senior_users ADD COLUMN IF NOT EXISTS profile_life_events JSONB;
        ALTER TABLE senior_users ADD COLUMN IF NOT EXISTS last_contact_ts TIMESTAMPTZ;
        ```

2.  **初始化 Milvus 知識庫**:
    在已啟動虛擬環境的終端機中，執行知識庫載入腳本。
    ```bash
    python load_article.py
    ```
    *(`user_memory` collection 將在第一次對話結束時自動建立)*

### 四、運行與測試

#### 4.1 啟動所有服務

您需要開啟 **三個** 獨立的終端機視窗，並確保它們都已進入 Poetry 虛擬環境 (`poetry shell`)。

* **終端機 1: 啟動 ngrok 通道**
    ```bash
    ngrok http 5000
    ```
    複製 `https://...` 開頭的 Forwarding 網址，並將其填入 LINE Developers Console 的 Webhook URL 設定中 (結尾需加上 `/webhook`)。

* **終端機 2: 啟動聊天機器人 Web 服務**
    ```bash
    python main.py
    ```

* **終端機 3: 啟動主動關懷排程服務**
    ```bash
    python ProactiveCare/scheduler.py
    ```

#### 4.2 進行端對端測試

1.  **即時對話**: 用手機 LINE App 將您的機器人加為好友，開始進行多輪對話，驗證 STM 的連續性。
2.  **LTM 摘要**: 閒置 5 分鐘或手動結束 `main.py` (`Ctrl+C`)，觸發 `finalize_session`。接著執行 `python view_memory_collection.py` 來驗證 LTM 是否已成功生成並存入 Milvus。
3.  **主動關懷**: 依照測試指南，在 DBeaver 中手動修改 `last_contact_ts`，觀察 `scheduler.py` 的日誌，並在您的手機上接收主動關懷訊息。
4.  **流程銜接**: 回覆該則主動關懷訊息，觀察 `main.py` 是否能無縫接續對話。

---

## 核心檔案結構說明

* `main.py`: **聊天機器人主程式**。一個 Flask Web 應用，負責接收 LINE Webhook，並協調即時回應流程。
* `ProactiveCare/scheduler.py`: **主動關懷排程器**。一個獨立的常駐程序，使用 APScheduler 管理所有主動關懷任務。
* `ProactiveCare/tasks.py`: **主動關懷任務邏輯**。定義了如何查詢使用者、組合上下文、生成並發送關懷訊息的核心業務邏輯。
* `HealthBot/agent.py`: **AI 智慧核心**。定義了 `Guardrail`, `Companion` 等 Agent 的人格與目標，並封裝了記憶生成 (`refine_summary`) 和情境建構 (`build_prompt_from_redis`) 的關鍵函式。
* `toolkits/redis_store.py`: **狀態管理中心**。封裝了所有對 Redis 的原子性讀寫操作，是確保系統併發安全和狀態一致的基石。
* `toolkits/tools.py`: **Agent 的工具箱**。提供了 `SearchMilvusTool` (衛教 RAG) 等 Agent 在執行任務時可以呼叫的具體能力。
* `utils/`: **共用輔助模組**。包含資料庫連線 (`db_connectors.py`) 和 LINE 推送 (`line_pusher.py`) 等可被多個程序複用的功能。
* `docker-compose.yml`: **基礎設施即程式碼**。定義並管理專案所需的所有後端服務 (PostgreSQL, Redis, Milvus)。