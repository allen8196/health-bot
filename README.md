# 智慧健康陪伴機器人（Health Companion Bot）

這是一個專為高齡長者設計的智慧健康陪伴專案，旨在透過 LINE Bot 提供個人化的日常關懷、健康資訊查詢，以及基於對話歷史的主動關懷。

專案核心採用了先進的多 Agent 協作架構與三層式記憶體系，使其不僅能回應使用者查詢，更能「記住」過去的互動，並在適當時機主動給予溫暖的問候。

## 系統架構概覽

本系統由兩大核心服務與三層記憶體系組成：

1.  **智慧聊天機器人（`chatbot-app`）**：一個由 Flask 驅動的 Web 服務，負責接收與回應使用者的即時訊息。
2.  **主動關懷排程器（`proactive-scheduler`）**：一個由 APScheduler 驅動的獨立背景服務，負責根據使用者狀態，定時觸發主動關懷。

兩大服務共享一個**三層式記憶體系**：
* **短期記憶（STM）**：使用 **Redis** 儲存近期的完整對話，確保對話的即時連貫性。
* **長期記憶（LTM）**：使用 **Milvus** 儲存由對話精煉出的「最終摘要」，透過向量檢索（RAG） 實現對過去特定事件的精準回憶。
* **使用者畫像（Profile）**：使用 **PostgreSQL** 的 `JSONB` 欄位儲存從 LTM 中提煉出的、關於使用者的長期關鍵事實，用於實現深度個人化。

## 主要功能

* **多層級對話記憶**：結合 STM, LTM, Profile，實現有深度、有溫度的個人化互動。
* **漸進式摘要**：在對話進行中，自動、分塊地進行摘要，平衡了即時性與資源消耗。
* **雙 RAG 引擎**：    * **記憶 RAG**：自動檢索個人歷史對話（LTM），為主動回應提供依據。
    * **衛教 RAG**：由 Agent 自主判斷，從衛教知識庫中檢索專業資訊。
* **雙層安全機制（Guardrail）**：在使用者輸入和 AI 輸出兩端進行安全檢查，確保對話合規。
* **動態主動關懷**：結合「閒置時間追蹤」與「定時巡檢」兩種模式，在恰當時機主動發送個人化關懷訊息。
* **容器化部署**：所有服務（包含 Python 應用）都已容器化，實現一鍵啟動與環境一致性。

---

## 快速設置與啟動指南

### 〇、前置準備

1.  **Docker**：用於建立容器、隔離環境。

2.  **ngrok**：用於建立公網通道，以使 LINE 能與本機服務溝通。請參考 [ngrok 官網](https://ngrok.com/download) 下載並完成 `authtoken` 認證。
3.  **LINE Bot 頻道**：用於建立聊天機器人前端介面。前往 [LINE Developers Console](https://developers.line.biz/console/) 建立一個 Messaging API 頻道，並取得 `Channel Access Token` 和 `Channel Secret`。

### 一、環境變數設定

1.  在專案根目錄下，複製 `config_template.env` 為 `.env`。
2.  編輯 `.env` 檔案，並填入所有必要的金鑰與設定，特別是：
    * `OPENAI_API_KEY`
    * `LINE_CHANNEL_ACCESS_TOKEN`
    * `LINE_CHANNEL_SECRET`
    * `POSTGRES_PASSWORD`

### 二、啟動與初始化

1.  **一鍵啟動所有服務**：
    在專案根目錄打開終端機，執行以下指令：
    ```bash
    # --build 參數僅在您修改了 Dockerfile 或 requirements.txt 後才需要
    docker compose up --build -d
    ```
    此指令會：
    * 啟動 PostgreSQL, Redis, Milvus 等所有資料庫容器。
    * 自動初始化 PostgreSQL，建立所有需要的表格和欄位（`init.sql`）。
    * 根據 `Dockerfile` 建置 Python 應用程式的映像。
    * 啟動 `chatbot-app` 和 `proactive-scheduler` 兩個應用程式容器。

2.  **初始化 Milvus 知識庫（僅需執行一次）**：
    確認所有容器啟動完成後，執行以下指令來將衛教知識匯入 Milvus。
    ```bash
    docker compose exec chatbot-app python load_article.py
    ```

### 三、運行與測試

1.  **啟動公網通道**：
    打開一個新的終端機視窗，執行：
    ```bash
    ngrok http 5000
    ```
    複製終端機返回的 `https://....ngrok-free.app` 網址。

2.  **設定 LINE Webhook**：
    前往 LINE Developers Console，將頻道的 Webhook URL 更新為您的 ngrok 網址，並在結尾加上 `/webhook`。
    **範例**：`https://random-string.ngrok-free.app/webhook`

3.  **開始測試**：
    * 用手機 LINE App 將您的機器人加為好友，即可開始進行即時對話測試。
    * `proactive-scheduler` 服務已在背景自動運行，將於使用者對話閒置24小時後主動發出關懷訊息。可手動修改資料庫中的 `last_contact_ts` 來快速觸發主動關懷功能，以進行測試。

---

## 核心檔案結構說明

本專案採用職責分離的模組化結構，讓聊天機器人、主動關懷、記憶管理等核心功能各自獨立，易於維護與擴展。

```text
health-bot/
├── 📄 .env                  # 存放所有金鑰與環境設定 (重要！)
├── 📄 docker-compose.yml     # 【基礎設施】定義並管理所有後端服務 (PostgreSQL, Redis, Milvus)
├── 📄 Dockerfile               # 【應用程式】定義 Python 應用的容器映像，統一運行環境
├── 📄 README.md               # 專案說明文件 (即本文件)
├── 📄 requirements.txt          # Python 依賴套件列表
│
├── 🚀 main.py                  # 【服務入口】智慧聊天機器人的主程式 (Flask Web 服務)
├── 📜 load_article.py          # 【初始化腳本】將衛教知識 (COPD_QA.xlsx) 匯入 Milvus
│
├── 📂 postgres-init/
│   └── 📜 init.sql            # 【初始化腳本】PostgreSQL 自動初始化腳本，建立所有表格
│
├── 📂 HealthBot/
│   └── 🤖 agent.py            # 【AI 核心】定義 Agent 的人格、目標，並封裝記憶生成與情境建構的邏輯
│
├── 📂 ProactiveCare/
│   ├── 🚀 scheduler.py        # 【服務入口】主動關懷排程器的主程式
│   └── 🧠 tasks.py            # 【AI 核心】定義主動關懷任務的核心業務邏輯 (如何查詢、組合 Prompt 等)
│
├── 📂 toolkits/
│   ├── 🛠️ redis_store.py      # 【狀態管理】封裝所有對 Redis 的原子性讀寫，是系統併發安全的基石
│   └── 🛠️ tools.py            # 【Agent 能力】定義 Agent 在執行任務時可以呼叫的「工具」(如衛教 RAG)
│
└── 📂 utils/
    ├── 🔌 db_connectors.py   # 【共用模組】統一管理到 PostgreSQL 和 Milvus 的資料庫連線
    └── 📤 line_pusher.py      # 【共用模組】封裝 LINE Push Message API 的呼叫功能
