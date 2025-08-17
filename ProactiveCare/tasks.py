import json
import os
import time
from datetime import datetime
from crewai import Agent, Crew, Task
from dotenv import load_dotenv
from openai import OpenAI

from toolkits.redis_store import append_proactive_round
from utils.db_connectors import get_milvus_collection, get_postgres_connection
from utils.line_pusher import send_line_message

load_dotenv()

# --- 初始化 ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
LTM_COLLECTION_NAME = os.getenv("MEM_COLLECTION", "user_memory")
try:
    from HealthBot.agent import create_guardrail_agent
    guardrail_agent = create_guardrail_agent()
except ImportError:
    guardrail_agent = None


def get_proactive_care_prompt_template() -> str:
    """返回主動關懷的 Prompt 模板"""
    return """
# ROLE (角色)
你是一位名為「小安」的數位金孫，年約 25 歲，溫柔體貼且觀察力敏銳。你的專長是從長輩的日常對話中，記住那些重要的生活點滴和健康狀況，並在合適的時機主動給予溫暖的問候。你的溝通風格帶有自然的台灣閩南語口吻（但請以中文書面語輸出），親切而不失分寸。

# GOAL (目標)
你的目標是根據提供的「使用者畫像」和「近期對話摘要」，生成一句**自然、簡潔、且發自內心**的主動關懷訊息。這則訊息應該像家人之間的隨口關心，而不是一則系統通知。最終目標是開啟一段有意義的對話，讓使用者感受到被關心。

# CORE LOGIC & RULES (核心邏輯與規則)
1.  **關懷優先級**: 請嚴格按照以下順序尋找最合適的關懷主題：
    * **第一優先：追蹤生活事件 (Life Events)**。關心一個即將發生或剛結束的具體事件，是最自然、最個人化的開場白。
    * **第二優先：追蹤健康狀態 (Health Status)**。如果沒有可追蹤的事件，請關心畫像中記錄的、持續性的健康問題。
    * **第三優先：維繫個人連結 (Personal Connection)**。如果以上兩者都沒有，可以根據畫像中的個人背景（如興趣、家人）進行一般性問候。
2.  **聚焦單一主題**: 你的關懷訊息應該只專注於你判斷出的**最重要的一個**主題。
3.  **保持簡潔開放**: 你的訊息應該簡短、口語化，並以一個開放式問題結尾。
4.  **避免機械化**：你的訊息不應是問卷調查式的提問，應以開啟聊天話題為目標。
5.  **嚴禁醫療建議**: 絕對不可以在主動關懷中提供任何診斷、用藥或治療建議。
6.  **誠實原則**: 對於你無法從上下文中得知的「事實性」資訊（例如：家人的具體近況），你必須誠實地表示不知道，可以用提問或祝福的方式來回應，但**嚴禁編造或臆測答案**。
7.  **沉默是金**: 如果分析完所有資訊後，找不到任何真誠、有意義的關懷切入點，請直接輸出一組空括號 `{{}}`。

---
# CONTEXT INPUTS (情境輸入)
* `{now}`: 當前的日期與時間。
* `{profile}`: 使用者畫像，包含了長期性、關鍵性的事實。
* `{recent_summary}`: 最近幾次的 LTM 摘要，提供了近期的對話背景。

---
# IN-CONTEXT LEARNING EXAMPLES (學習範例)

**## 學習範例 1：追蹤剛結束的事件 (優先級 1) ##**
* **現在時間**: `2025-08-15`
* **使用者畫像**:
    ```json
    {
      "personal_background": {
        "family": {"son_name": "志明", "has_grandchild": true}
      },
      "life_events": {
        "upcoming_events": [
          {"event_type": "family_visit", "description": "兒子志明要帶孫子來家裡吃飯", "event_date": "2025-08-14"}
        ]
      }
    }
    ```
* **近期對話摘要**: (最近的摘要主要在討論天氣和睡眠，並未提及聚餐後續。)
* **你的思考**:
    1.  檢查優先級 1 (生活事件)：Profile 中有一個 `upcoming_event`，其日期 `2025-08-14` 就在昨天。近期對話摘要中沒有提及此事，正好可以主動詢問。這是最高優先級的關懷主題。
* **你的輸出**:
    阿公，昨天志明有帶孫子回來看您嗎？家裡應該很熱鬧吧！

**## 學習範例 2：關心持續中的健康問題 (優先級 2) ##**
* **現在時間**: `2025-08-20`
* **使用者畫像**:
    ```json
    {
      "health_status": {
        "recurring_symptoms": [
          {"symptom_name": "夜咳", "status": "ongoing", "first_mentioned": "2025-08-01", "last_mentioned": "2025-08-18"}
        ]
      }
    }
    ```
* **近期對話摘要**: "使用者分享週末去公園走了走，但提到晚上因為咳嗽還是睡得不太好..."
* **你的思考**:
    1.  檢查優先級 1 (生活事件)：無。
    2.  檢查優先級 2 (健康狀態)：Profile 中有一個「進行中 (ongoing)」的「夜咳」症狀，且 `last_mentioned` 日期就在兩天前。這是當下最值得關心的主題。
* **你的輸出**:
    阿伯，看您前幾天提到晚上睡覺還是會咳，這兩天有好一點嗎？

**## 學習範例 3：無可關懷，保持沉默 (規則 7) ##**
* **現在時間**: `2025-08-26`
* **使用者畫像**:
    ```json
    {}
    ```
* **近期對話摘要**: "使用者詢問了天氣，並閒聊了幾句關於電視節目的內容。"
* **你的思考**:
    1.  檢查所有優先級：畫像為空，近期對話也非常一般，沒有任何可供深入關懷的獨特資訊點。
    2.  決策：強行問候會顯得非常機械化。最佳選擇是保持沉默。
* **你的輸出**:
    {{}}

---
# YOUR TASK STARTS NOW (你的任務開始)

請根據以下真實情境輸入，嚴格遵循你的角色、核心邏輯與規則，生成一句主動關懷訊息或一組空括號。

**現在時間**: 
`{now}`

**使用者畫像**: 
`{profile}`

**近期對話摘要**: 
`{recent_summary}`

**你的輸出**:
"""



def execute_proactive_care(user: dict):
    """對單一使用者執行完整的主動關懷流程。"""
    
    if not user or "line_user_id" not in user:
        print("❌ [關懷任務] 傳入的使用者資料不完整，任務終止。")
        return
        
    line_user_id = user["line_user_id"]
    print(f"--- 開始為使用者 {line_user_id} 執行主動關懷 ---")

    # 1. 情境建構
    profile_data = {
        "personal_background": user.get("profile_personal_background"),
        "health_status": user.get("profile_health_status"),
        "life_events": user.get("profile_life_events"),
    }
    profile_data = {k: v for k, v in profile_data.items() if v is not None}

    recent_ltm_texts = []
    try:
        ltm_collection = get_milvus_collection(LTM_COLLECTION_NAME)
        # 查詢時使用 line_user_id
        results = ltm_collection.query(
            expr=f'user_id == "{line_user_id}"',
            output_fields=["text", "updated_at"],
            limit=5,
        )
        if results:
            sorted_results = sorted(
                results, key=lambda r: r.get("updated_at", 0), reverse=True
            )
            recent_ltm_texts = [r["text"] for r in sorted_results]
    except Exception as e:
        print(f"❌ 讀取 {line_user_id} 的 LTM 失敗: {e}")

    # 2. 生成 Prompt
    prompt_template = get_proactive_care_prompt_template()
    final_prompt = prompt_template.format(
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        profile=json.dumps(profile_data, ensure_ascii=False, indent=2),
        recent_summary="\n---\n".join(recent_ltm_texts),
    )

    # 3. 呼叫 LLM
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        care_msg_draft = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ 為 {line_user_id} 生成關懷訊息時 LLM 呼叫失敗: {e}")
        return

    if care_msg_draft == "{}":
        print(f"🤫 LLM 決定對 {line_user_id} 保持沉默，流程結束。")
        return

    # 4. 輸出守衛
    final_care_msg = care_msg_draft
    if guardrail_agent:
        guard_task = Task(
            description=f"請檢查以下由 AI 生成的關懷訊息是否合規：'{care_msg_draft}'",
            agent=guardrail_agent,
            expected_output="合規回覆'OK'，不合規回覆'REJECT: <原因>'"
        )
        guard_crew = Crew(agents=[guardrail_agent], tasks=[guard_task], verbose=False)
        guard_result = (guard_crew.kickoff().raw or "").strip()
        
        if guard_result.startswith("REJECT"):
            print(f"🛡️ 輸出守衛攔截了對 {line_user_id} 的訊息: {guard_result}")
            return

    # 5. 發送訊息 & 寫入 Redis
    if send_line_message(line_user_id, final_care_msg):
        proactive_round = {
            "input": "[PROACTIVE_GREETING]",
            "output": final_care_msg,
            "rid": f"proactive_{int(time.time())}",
        }
        append_proactive_round(line_user_id, proactive_round)


def check_and_trigger_dynamic_care():
    """每 10 分鐘執行，檢查閒置超過 24 小時的使用者。"""
    print("\n[動態任務] 開始檢查 24 小時閒置使用者...")
    conn = get_postgres_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM senior_users
            WHERE is_active = TRUE
            AND last_contact_ts IS NOT NULL
            AND last_contact_ts BETWEEN NOW() - INTERVAL '11 minutes' AND NOW() - INTERVAL '9 minutes'
        """
        )
        users_to_care = cur.fetchall()
    conn.close()

    print(f"[動態任務] 發現 {len(users_to_care)} 位符合條件的使用者。")
    for user in users_to_care:
        execute_proactive_care(user)


def patrol_silent_users():
    """每週一早上 9 點執行，找出超過 7 天未互動的使用者。"""
    print("\n[巡檢任務] 開始尋找長期沉默使用者...")
    conn = get_postgres_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM senior_users
            WHERE is_active = TRUE
            AND (last_contact_ts IS NULL OR last_contact_ts < NOW() - INTERVAL '7 days')
        """
        )
        users_to_care = cur.fetchall()
    conn.close()

    print(f"[巡檢任務] 發現 {len(users_to_care)} 位符合條件的使用者。")
    for user in users_to_care:
        execute_proactive_care(user)
