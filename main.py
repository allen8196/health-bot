import hashlib
import os
import threading
import time
from typing import Optional

from crewai import Crew, Task
from flask import Flask, abort, request
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from pymilvus import connections

from HealthBot.agent import (
    build_prompt_from_redis,
    create_guardrail_agent,
    create_health_companion,
    finalize_session,
)
from toolkits.redis_store import (
    append_audio_segment,
    append_round,
    get_audio_result,
    get_redis,
    make_request_id,
    peek_next_n,
    read_and_clear_audio_segments,
    set_audio_result,
    set_state_if,
    try_register_request,
    xadd_alert,
)
from toolkits.tools import summarize_chunk_and_commit

# Flask App 初始化
app = Flask(__name__)

# LINE Bot SDK 初始化
line_config = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN", ""))
line_handler = WebhookHandler(
    os.getenv("LINE_CHANNEL_SECRET", "")
)  # 請在 .env 和 LINE Console 中補上 Channel Secret

SUMMARY_CHUNK_SIZE = int(os.getenv("SUMMARY_CHUNK_SIZE", 5))


class AgentManager:
    def __init__(self):
        self.guardrail_agent = create_guardrail_agent()
        self.health_agent_cache = {}

    def get_guardrail(self):
        return self.guardrail_agent

    def get_health_agent(self, user_id: str):
        if user_id not in self.health_agent_cache:
            self.health_agent_cache[user_id] = create_health_companion(user_id)
        return self.health_agent_cache[user_id]

    def release_health_agent(self, user_id: str):
        if user_id in self.health_agent_cache:
            del self.health_agent_cache[user_id]


# ---- Persist & maybe summarize ----


def log_session(user_id: str, query: str, reply: str, request_id: Optional[str] = None):
    rid = request_id or make_request_id(user_id, query)
    if not try_register_request(user_id, rid):
        print("[去重] 跳過重複請求")
        return
    append_round(user_id, {"input": query, "output": reply, "rid": rid})
    # 嘗試抓下一段 5 輪（不足會回空）→ LLM 摘要 → CAS 提交
    start, chunk = peek_next_n(user_id, SUMMARY_CHUNK_SIZE)
    if start is not None and chunk:
        summarize_chunk_and_commit(user_id, start_round=start, history_chunk=chunk)


# ---- Pipeline ----


def handle_user_message(
    agent_manager: AgentManager,
    user_id: str,
    query: str,
    audio_id: Optional[str] = None,
    is_final: bool = True,
) -> str:
    # 0) 統一音檔 ID（沒帶就用文字 hash 當臨時 ID，向後相容）
    audio_id = audio_id or hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]

    # 1) 非 final：不觸發任何 LLM/RAG/通報，只緩衝片段
    if not is_final:
        append_audio_segment(user_id, audio_id, query)
        return "👌 已收到語音片段"

    # 2) 音檔級鎖：一次且只一次處理同一段音檔
    lock_id = f"{user_id}#audio:{audio_id}"
    if not set_state_if(lock_id, expect="", to="PROCESSING"):
        # 可能已處理或處理中 → 回快取或提示
        cached = get_audio_result(user_id, audio_id)
        return cached or "我正在處理你的語音，請稍等一下喔。"

    try:
        # 3) 合併之前緩衝的 partial → 最終要處理的全文
        head = read_and_clear_audio_segments(user_id, audio_id)
        full_text = (head + " " + query).strip() if head else query

        # 4) 原本流程：先 guardrail，再 health agent（你現有碼原封搬過來）
        # 設置環境變數供工具使用
        os.environ["CURRENT_USER_ID"] = user_id

        guard = agent_manager.get_guardrail()
        guard_task = Task(
            description=(
                f"判斷是否需要攔截：「{full_text}」。"
                "務必使用 model_guardrail 工具進行判斷；"
                "安全回 OK；需要攔截時回 BLOCK: <原因>（僅此兩種）。"
            ),
            expected_output="OK 或 BLOCK: <原因>",
            agent=guard,
        )
        guard_res = (
            Crew(agents=[guard], tasks=[guard_task], verbose=False).kickoff().raw or ""
        ).strip()
        if guard_res.startswith("BLOCK:"):
            reason = guard_res[6:].strip()
            # 檢查是否涉及自傷風險，需要通報個管師
            if any(k in reason for k in ["自傷", "自殺", "傷害自己", "緊急"]):
                xadd_alert(
                    user_id=user_id,
                    reason=f"可能自傷風險：{full_text}",
                    severity="high",
                )
            reply = "抱歉，這個問題涉及違規或需專業人士評估，我無法提供解答。"
            set_audio_result(user_id, audio_id, reply)
            log_session(user_id, full_text, reply)
            return reply

        care = agent_manager.get_health_agent(user_id)
        ctx = build_prompt_from_redis(user_id, k=6, current_input=full_text)
        task = Task(
            description=f"{ctx}\n\n使用者輸入：{full_text}\n請以台語風格溫暖務實回覆；有需要查看COPD相關資料或緊急事件需要通報時，請使用工具。",
            expected_output="台語風格的溫暖關懷回覆，必要時使用工具。",
            agent=care,
        )
        res = Crew(agents=[care], tasks=[task], verbose=False).kickoff().raw or ""

        # 5) 結果快取 + 落歷史（你原本就有 log_session）
        set_audio_result(user_id, audio_id, res)
        log_session(user_id, full_text, res)
        return res

    finally:
        set_state_if(lock_id, expect="PROCESSING", to="FINALIZED")


class UserSession:
    def __init__(self, user_id: str, agent_manager: AgentManager, timeout: int = 300):
        self.user_id = user_id
        self.agent_manager = agent_manager
        self.timeout = timeout
        self.last_active_time = None
        self.stop_event = threading.Event()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def update_activity(self):
        self.last_active_time = time.time()

    def _watchdog(self):
        while not self.stop_event.is_set():
            time.sleep(5)
            if self.last_active_time and (
                time.time() - self.last_active_time > self.timeout
            ):
                print(f"\n⏳ 閒置超過 {self.timeout}s，開始收尾...")
                finalize_session(self.user_id)
                self.agent_manager.release_health_agent(self.user_id)
                self.stop_event.set()


agent_manager = AgentManager()
session_pool = {}


# --- Flask Webhook 端點 ---
@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


@line_handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_id = event.source.user_id
    query = event.message.text

    print(f"收到來自 {user_id} 的訊息: {query}")

    # 確保每個使用者都有一個 session
    if user_id not in session_pool:
        session_pool[user_id] = UserSession(user_id, agent_manager)

    session = session_pool[user_id]
    session.update_activity()  # 更新活動時間

    # 呼叫您現有的核心處理邏輯
    reply_text = handle_user_message(agent_manager, user_id, query)

    # 使用 LINE SDK 回覆訊息
    with ApiClient(line_config) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token, messages=[TextMessage(text=reply_text)]
            )
        )


def run_app():
    # 啟動 Flask 應用
    # 注意：在生產環境中應使用 Gunicorn 或其他 WSGI 伺服器
    app.run(port=5000, debug=True, use_reloader=False)


def main() -> None:
    connections.connect(
        alias="default", uri=os.getenv("MILVUS_URI", "http://localhost:19530")
    )
    am = AgentManager()
    uid = os.getenv("TEST_USER_ID", "test_user")
    sess = UserSession(uid, am)
    print("✅ 啟動完成，閒置 5 分鐘：補分段摘要→Refine→Purge")
    try:
        am.get_health_agent(uid)
        while not sess.stop_event.is_set():
            try:
                q = input("🧓 長輩：").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not q:
                continue
            sess.update_activity()
            a = handle_user_message(am, uid, q)
            print("👧 金孫：", a)
    finally:
        if not sess.stop_event.is_set():
            print("\n📝 結束對話：收尾...")
            finalize_session(uid)
        am.release_health_agent(uid)
        print("👋 系統已關閉")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    run_app()
