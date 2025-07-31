import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import Collection, connections
from embedding import to_vector
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# === 初始化 ===
load_dotenv()
model_name = "gpt-4o-mini"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD"))
chat_model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name)
llm_api = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 狀態管理 ===
def load_user_context(user_id: str) -> dict:
    os.makedirs("sessions", exist_ok=True)
    os.makedirs("profiles", exist_ok=True)
    summary_path = f"sessions/{user_id}_summary.json"
    profile_path = f"profiles/{user_id}.json"

    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"summary": ""}, f)

    if not os.path.exists(profile_path):
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump({"age": None, "personality": "溫和"}, f)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f).get("summary", "")
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    return {"summary": summary, "profile": profile}

# === Tool 1: RAG 查詢 ===
@tool
def search_milvus(query: str) -> str:
    """在 Milvus 資料庫中查詢 COPD 衛教問答，回傳相似問題與答案"""
    try:
        connections.connect(alias="default", uri="http://localhost:19530")
        collection = Collection("copd_qa")
        collection.load()
        user_vec = to_vector(query)
        if not isinstance(user_vec, list):
            user_vec = user_vec.tolist() if hasattr(user_vec, 'tolist') else list(user_vec)
        results = collection.search(
            data=[user_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=5,
            output_fields=["question", "answer", "category"],
        )
        connections.disconnect(alias="default")

        chunks = []
        for hit in results[0]:
            if hit.score >= SIMILARITY_THRESHOLD:
                q = hit.entity.get("question")
                a = hit.entity.get("answer")
                cat = hit.entity.get("category")
                chunks.append(f"[{cat}] (相似度: {hit.score:.3f})\nQ: {q}\nA: {a}")

        return "\n\n".join(chunks) if chunks else "[查無高相似度結果]"
    except Exception as e:
        return f"[Milvus 錯誤] {e}"

# === Tool 2: 對話摘要 ===
@tool
def summarize_conversation(user_id: str) -> str:
    """摘要整段對話紀錄，並更新使用者的摘要檔案"""
    session_path = f"sessions/{user_id}.json"
    summary_path = f"sessions/{user_id}_summary.json"
    if not os.path.exists(session_path):
        return "目前無可供摘要的對話紀錄。"

    with open(session_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    text = "".join([f"第{i+1}輪:\n長輩: {h['input']}\n金孫: {h['output']}\n\n" for i, h in enumerate(history)])
    prompt = f"""
請為以下對話生成摘要，涵蓋健康問題、建議重點、情緒氛圍：\n{text}請用繁體中文回答，100-150字。
"""

    try:
        res = llm_api.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是摘要助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        summary = res.choices[0].message.content.strip()
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return summary
    except Exception as e:
        return f"[摘要錯誤] {e}"

# === 建立聊天 Agent ===
def build_agent(user_id: str):
    context = load_user_context(user_id)
    profile = context["profile"]
    summary = context["summary"]

    profile_txt = f"使用者年齡：{profile.get('age', '未知')}，個性：{profile.get('personality', '溫和')}\n"
    summary_txt = f"\n\n📌 歷史摘要：\n{summary}" if summary else ""

    system_msg = f"""
你是一位會說台灣閩南語的健康陪伴機器人。
{profile_txt}
你可以使用 search_milvus 查詢健康知識庫，或使用 summarize_conversation 來總結最近的對話。
請根據需要決定是否使用這些工具。請以親切台語進行對話。{summary_txt}
""".strip()

    tools = [search_milvus, summarize_conversation]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=chat_model,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=memory,
        agent_kwargs={"system_message": system_msg}
    )
    return agent

# === 離線前自動摘要 ===
def auto_save_and_summary(user_id: str):
    print("📝 自動儲存並進行對話摘要中...")
    print(summarize_conversation.invoke({"user_id": user_id}))
