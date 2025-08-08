from datetime import datetime
from crewai import Agent
from toolkits.tools import SearchMilvusTool, AlertCaseManagerTool, RiskKeywordCheckTool, summarize_chunk_and_commit
from toolkits.redis_store import fetch_unsummarized_tail, fetch_all_history, get_summary, peek_next_n, peek_remaining, set_state_if, purge_user_session
from openai import OpenAI
import os, json

STM_MAX_CHARS = int(os.getenv("STM_MAX_CHARS", 1800))
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", 3000))
REFINE_CHUNK_ROUNDS = int(os.getenv("REFINE_CHUNK_ROUNDS", 20))
SUMMARY_CHUNK_SIZE = int(os.getenv("SUMMARY_CHUNK_SIZE", 5))

class UserProfileManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.profile_path = f"profiles/{user_id}.json"
        os.makedirs("profiles", exist_ok=True)
        if not os.path.exists(self.profile_path):
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump({"age": None, "personality": "溫和", "refined_summary": ""}, f, ensure_ascii=False, indent=2)

    def load_profile(self) -> dict:
        with open(self.profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "refined_summary" not in data:
            data["refined_summary"] = ""; self.save_profile(data)
        return data

    def save_profile(self, profile: dict):
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    def update_refined_summary(self, text: str) -> None:
        prof = self.load_profile(); model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        tag = f"--- {datetime.now().strftime('%Y-%m-%d')} ({model}) 更新 ---"
        prof["refined_summary"] = (prof.get("refined_summary","" ) + ("\n\n" if prof.get("refined_summary") else "") + tag + "\n" + text.strip() + "\n\n")
        self.save_profile(prof)

# ---- Prompt 構建 ----

def _shrink_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars: return text
    tail = text[-max_chars:]; idx = tail.find("--- ")
    return tail[idx:] if idx != -1 else tail

def build_prompt_from_redis(user_id: str, k: int = 6) -> str:
    summary, _ = get_summary(user_id)
    summary = _shrink_tail(summary, SUMMARY_MAX_CHARS) if summary else ""
    rounds = fetch_unsummarized_tail(user_id, k=max(k,1))
    def render(rs): return "\n".join([f"長輩：{r['input']}\n金孫：{r['output']}" for r in rs])
    chat = render(rounds)
    while len(chat) > STM_MAX_CHARS and len(rounds) > 1:
        rounds = rounds[1:]; chat = render(rounds)
    if len(chat) > STM_MAX_CHARS and rounds: chat = chat[-STM_MAX_CHARS:]
    prof = UserProfileManager(user_id).load_profile()
    parts = [f"使用者年齡：{prof.get('age','未知')}，個性：{prof.get('personality','溫和')}"]
    if summary: parts.append("📌 歷史摘要：\n" + summary)
    if prof.get('refined_summary'): parts.append("⭐ 長期追蹤重點：\n" + prof['refined_summary'])
    if chat: parts.append("🕓 近期對話（未摘要）：\n" + chat)
    return "\n\n".join(parts)

# ---- Agents ----

def create_guardrail_agent() -> Agent:
    return Agent(role="風險檢查員", goal="攔截危險/違法/自傷內容", backstory="你是系統第一道安全防線。", tools=[RiskKeywordCheckTool()], memory=False, verbose=False)

def create_health_companion(user_id: str) -> Agent:
    return Agent(role="健康陪伴者", goal="以台語關懷長者健康與心理狀況，必要時通報", backstory="你是會講台語的金孫型陪伴機器人，回覆溫暖務實。", tools=[SearchMilvusTool(), AlertCaseManagerTool()], memory=False, verbose=False)

# ---- Refine（map-reduce over 全量 QA） ----

def refine_summary(user_id: str) -> None:
    all_rounds = fetch_all_history(user_id)
    if not all_rounds: return
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # 1) 分片
    chunks = [all_rounds[i:i+REFINE_CHUNK_ROUNDS] for i in range(0, len(all_rounds), REFINE_CHUNK_ROUNDS)]
    partials = []
    for ch in chunks:
        conv = "\n".join([f"第{i+1}輪\n長輩:{c['input']}\n金孫:{c['output']}" for i,c in enumerate(ch)])
        res = client.chat.completions.create(
            model=os.getenv("MODEL_NAME","gpt-4o-mini"), temperature=0.3,
            messages=[{"role":"system","content":"你是專業的健康對話摘要助手。"},{"role":"user","content":f"請摘要成 80-120 字（病況/情緒/生活/建議）：\n\n{conv}"}],
        )
        partials.append((res.choices[0].message.content or "").strip())
    comb = "\n".join([f"• {s}" for s in partials])
    res2 = client.chat.completions.create(
        model=os.getenv("MODEL_NAME","gpt-4o-mini"), temperature=0.4,
        messages=[{"role":"system","content":"你是臨床心理與健康管理顧問。"},{"role":"user","content":f"整合以下多段摘要為不超過 180 字、條列式精緻摘要（每行以 • 開頭）：\n\n{comb}"}],
    )
    final = (res2.choices[0].message.content or "").strip()
    UserProfileManager(user_id).update_refined_summary(final)

# ---- Finalize：補分段摘要 → Refine → Purge ----

def finalize_session(user_id: str) -> None:
    set_state_if(user_id, expect="ACTIVE", to="FINALIZING")
    start, remaining = peek_remaining(user_id)
    if remaining:
        summarize_chunk_and_commit(user_id, start_round=start, history_chunk=remaining)
    refine_summary(user_id)
    purge_user_session(user_id)