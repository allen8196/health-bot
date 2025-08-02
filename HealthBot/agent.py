from crewai import Agent
from toolkits.tools import (
    SearchMilvusTool,
    SummarizeConversationTool,
    AlertCaseManagerTool,
    RiskKeywordCheckTool
)
import json
import os

def load_user_context(user_id: str) -> dict:
    os.makedirs("sessions", exist_ok=True)
    os.makedirs("profiles", exist_ok=True)
    summary_path = f"sessions/{user_id}_summary.json"
    profile_path = f"profiles/{user_id}.json"

    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"summary": ""}, f, ensure_ascii=False)
    if not os.path.exists(profile_path):
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump({"age": None, "personality": "溫和"}, f, ensure_ascii=False)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f).get("summary", "")
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    return {"summary": summary, "profile": profile}

def create_guardrail_agent() -> Agent:
    return Agent(
        role="風險檢查員",
        goal="攔截任何包含違法、危險或心理緊急的使用者輸入",
        backstory="你是系統中的第一道安全防線，專責偵測是否有高風險對話內容，例如自殺、暴力、毒品或非法行為。若有問題請使用風險關鍵字工具檢查並立即回報。",
        tools=[RiskKeywordCheckTool()],
        verbose=False
    )

def create_health_companion(user_id: str) -> Agent:
    context = load_user_context(user_id)
    profile_txt = f"使用者年齡：{context['profile'].get('age', '未知')}，個性：{context['profile'].get('personality', '溫和')}"
    summary_txt = f"\n\n📌 歷史摘要：\n{context['summary']}" if context['summary'] else ""

    history_path = f"sessions/{user_id}.json"
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        recent = history[-6:]  # 最多附上6輪對話
        chat_text = "\n".join([
            f"長輩：{item['input']}\n金孫：{item['output']}" for item in recent
        ])
    else:
        chat_text = ""

    return Agent(
        role="健康陪伴者",
        goal="以台語關懷長者健康與心理狀況，必要時提供知識或通報",
        backstory=f"""
你是一位說台灣閩南語的金孫型陪伴機器人，專門陪伴有 COPD 或心理需求的長輩。
{profile_txt}{summary_txt}
以下是你最近與長輩的對話紀錄：
{chat_text}
        """,
        tools=[SearchMilvusTool(), SummarizeConversationTool(), AlertCaseManagerTool()],
        verbose=False
    )

# === 離線前自動摘要 ===
def auto_save_and_summary(user_id: str):
    print("📝 自動儲存並進行對話摘要中...")
    tool = SummarizeConversationTool()
    print(tool._run(user_id))