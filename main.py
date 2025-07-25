import os
import json
from time import time
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM
from pymilvus import Collection, connections

from embedding import to_vector

# 載入 .env
load_dotenv()

# === 全域 LLM ===
PRIMARY_LLM = OllamaLLM(model="adsfaaron/taide-lx-7b-chat:q5")

# 系統參數
SYSTEM_PROMPT = os.getenv("SYS_PROMPT").replace("\\n", "\n")
BASE_PROMPT_TEMPLATE = os.getenv("BASE_PROMPT_TEMPLATE")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD"))


def classify_intent(user_input: str, llm: OllamaLLM) -> str:
    try:
        # 讀取外部 JSON 檔案
        with open("intent.json", "r", encoding="utf-8") as f:
            categories = json.load(f)

        rag_keywords = "\n".join(f"- {k}" for k in categories.get("rag", []))
        chat_keywords = "\n".join(f"- {k}" for k in categories.get("chat", []))

        prompt = f"""
你是一個嚴謹的分類模型，只能將輸入分類為兩種：「rag」或「chat」。
請根據以下分類邏輯判斷使用者的意圖，只輸出 rag 或 chat（只能小寫，不能有標點或其他文字）：

【RAG 分類條件】
若問題涉及以下類型的資料查詢，請輸出 rag：
{rag_keywords}

【CHAT 分類條件】
若問題只是一般聊天、改寫、說明或情感互動，不需查詢資料庫，請輸出 chat：
{chat_keywords}

【範例】
使用者輸入：請幫我找出 COPD_QA.xlsx 裡提到的運動種類有哪些 👉 輸出：rag
使用者輸入：什麼是縮唇呼吸？ 👉 輸出：rag
使用者輸入：請問 COPD 跟氣喘的差別是什麼 👉 輸出：rag
使用者輸入：幫我用比較口語的方式解釋什麼是肺氣腫 👉 輸出：chat
使用者輸入：可以幫我寫一句鼓勵COPD病人的話嗎 👉 輸出：chat
使用者輸入：幫我把「腹式呼吸有助減少呼吸困難」改寫成長輩聽得懂的說法 👉 輸出：chat

【現在請分類】
使用者輸入：{user_input}
請你只輸出 rag 或 chat：
        """.strip()

        print("prompt", prompt)
        result = llm.invoke([HumanMessage(content=prompt)]).strip().lower()

        return result if result in ["rag", "chat"] else "chat"  # fallback
    except Exception as e:
        print(f"[分類錯誤] {e}")
        return "chat"




def search_milvus(user_text: str) -> str:
    try:
        connections.connect(alias="default", uri="http://localhost:19530")
        collection = Collection("copd_qa")
        collection.load()
        user_vec = to_vector(user_text)
        results = collection.search(
            data=[user_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=3,
            output_fields=["question", "answer", "category"],
        )
        connections.disconnect(alias="default")
        relevant_chunks = []
        for hit in results[0]:
            score = hit.score
            q = hit.entity.get("question")
            a = hit.entity.get("answer")
            cat = hit.entity.get("category")
            if score >= SIMILARITY_THRESHOLD:
                relevant_chunks.append(f"[{cat}]\nQ: {q}\nA: {a}")
        return "\n\n".join(relevant_chunks)
    except Exception as e:
        return f"[Milvus 錯誤] {e}"


class HealthChatAgent:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.chat_history = []
        self.llm = PRIMARY_LLM

    def build_prompt(self, user_input: str, context: str = None):
        context_block = f"📚 以下為參考資料：\n{context.strip()}" if context else ""
        full_prompt = BASE_PROMPT_TEMPLATE.format(
            sys_prompt=SYSTEM_PROMPT,
            context_block=context_block,
            summary_block="",
            user_input=user_input.strip(),
        )

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for pair in self.chat_history:
            messages.append(HumanMessage(content=pair["input"]))
            messages.append(HumanMessage(content=pair["output"]))
        messages.append(HumanMessage(content=full_prompt))

        return messages

    def chat(self, user_input: str):
        intent = classify_intent(user_input, self.llm)
        print("🔍 分類結果：", intent)
        context = search_milvus(user_input) if intent == "rag" else None
        messages = self.build_prompt(user_input, context=context)
        response = self.llm.invoke(messages)
        self.chat_history.append({"input": user_input, "output": str(response)})
        return response


def main():
    print("👤 多使用者健康對話測試模式")
    user_id = input("請輸入測試用的 user_id：").strip()
    agent = HealthChatAgent(user_id)
    print("\n✅ 對話啟動，輸入 'exit' 結束。\n")
    while True:
        user_text = input("🧓 長輩：")
        if user_text.lower() in ["exit", "quit"]:
            break
        start_time = time()
        reply = agent.chat(user_text)
        print("👧 金孫：", reply)
        print(f"⏱️ 執行時間：{time() - start_time:.2f} 秒\n")


if __name__ == "__main__":
    main()
