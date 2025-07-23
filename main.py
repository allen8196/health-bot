import os
from time import time

from dotenv import load_dotenv
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM
from pymilvus import Collection, connections

from embedding import to_vector

# 載入 .env
load_dotenv()

# 中文摘要 prompt
SUMMARY_PROMPT = """你是對話摘要助手，請用繁體中文將以下對話整理為簡潔摘要。
👓 先前摘要：
{summary}

💬 本輪對話：
{new_lines}

📝 請產生一段更新後的摘要：
"""


# System 人設
SYSTEM_PROMPT = os.getenv("SYS_PROMPT").replace("\\n", "\n")

# 明確描述知識庫範圍
KNOWLEDGE_BASE_SCOPE = os.getenv("KNOWLEDGE_BASE_SCOPE").replace("\\n", "\n")


class HealthChatAgent:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.llm = self._init_llm()
        self.memory = self._init_memory()

    def _init_llm(self):
        return OllamaLLM(model="adsfaaron/taide-lx-7b-chat:q5")

    def _init_memory(self):
        return ConversationSummaryMemory(
            llm=self.llm,
            memory_key="chat_history",
            prompt=ChatPromptTemplate.from_template(SUMMARY_PROMPT),
        )

    def search_milvus(self, user_text):
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
            threshold = float(os.getenv("SIMILARITY_THRESHOLD"))
            relevant_chunks = []
            print("\n🔍 前 3 筆相似 QA（含相似度）")
            for i, hit in enumerate(results[0]):
                score = hit.score
                q = hit.entity.get("question")
                a = hit.entity.get("answer")
                cat = hit.entity.get("category")
                print(f"Top {i+1} | 相似度: {score:.4f}\n[{cat}] Q: {q}\nA: {a}\n")
                if score >= threshold:
                    relevant_chunks.append(f"[{cat}]\nQ: {q}\nA: {a}")
            return relevant_chunks
        except Exception as e:
            print(f"[Milvus 錯誤] {e}")
            return []

    def intent_detect(self, user_input):
        light_llm = OllamaLLM(model="qwen:1.8b-chat")  # 小中文模型
        prompt = (
            f"{KNOWLEDGE_BASE_SCOPE}\n\n"
            f"以下是使用者的問題，請判斷是否需要查詢知識庫才能正確回答：\n"
            f"「{user_input}」\n\n"
            f"請只回答 yes 或 no，不要加其他文字。"
        )
        resp = light_llm.invoke([HumanMessage(prompt)])
        print("🤖 小模型意圖判斷結果：", resp)
        return "yes" in resp.lower()

    def build_prompt_by_template(self, user_input, context=None, summary=None):
        sys_prompt = SYSTEM_PROMPT
        base_template = os.getenv("BASE_PROMPT_TEMPLATE")

        context_block = f"📚 以下為參考資料：\n{context.strip()}" if context else ""
        summary_block = f"🧠 對話摘要：\n{summary.strip()}" if summary else ""

        full_prompt = base_template.format(
            sys_prompt=sys_prompt,
            context_block=context_block,
            summary_block=summary_block,
            user_input=user_input.strip(),
        )

        return [SystemMessage(content=sys_prompt), HumanMessage(content=full_prompt)]

    def chat(self, user_input):
        # === 1. 是否需查詢知識庫 ===
        need_rag = self.intent_detect(user_input)
        print("🤖 小模型意圖判斷結果：", need_rag)
        chunks = self.search_milvus(user_input) if need_rag else []
        context = "\n\n".join(chunks) if chunks else None

        # === 2. 載入摘要記憶（第二輪起才會有） ===
        history = self.memory.load_memory_variables({})["chat_history"]
        summary = history if history.strip() else None

        # === 3. 組裝 prompt ===
        chat_messages = self.build_prompt_by_template(
            user_input=user_input, context=context, summary=summary
        )

        # === 4. 取得回覆並記憶 ===
        response = self.llm.invoke(chat_messages)
        self.memory.save_context({"input": user_input}, {"output": response})

        print("\n🧠 使用者摘要記憶：")
        print(self.memory.load_memory_variables({})["chat_history"])

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
