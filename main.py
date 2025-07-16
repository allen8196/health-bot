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
SUMMARY_PROMPT = """請將以下使用者與助理的對話摘要為一段簡潔描述，摘要請使用繁體中文。

{summary}
對話內容：
{new_lines}
摘要："""

# System 人設
SYSTEM_PROMPT = "你是一位貼心的孫子/孫女，正在用自然、關懷的語氣和爺爺奶奶對話，請使用繁體中文，語氣親切簡短。不要講故事或過長的建議，請像日常對話一樣簡單回應。"


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
            collection = Collection("demo1")
            collection.load()
            user_vec = to_vector(user_text)
            results = collection.search(
                data=[user_vec],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=3,
                output_fields=["text"],
            )
            connections.disconnect(alias="default")

            threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
            relevant_chunks = []

            print("\n🔍 前 3 筆相似檢索結果（含相似度）")
            for i, hit in enumerate(results[0]):
                score = hit.score
                chunk_text = hit.entity.get("text")
                print(f"Top {i+1} | 相似度: {score:.4f}\n內容: {chunk_text}\n")
                if score >= threshold:
                    relevant_chunks.append(chunk_text)

            return relevant_chunks
        except Exception as e:
            print(f"[Milvus 錯誤] {e}")
            return []

    def chat(self, user_input):
        chunks = self.search_milvus(user_input)
        history = self.memory.load_memory_variables({})["chat_history"]

        context = "\n\n".join(chunks) if chunks else ""

        chat_messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if context:
            chat_messages.append(
                HumanMessage(content=f"以下是你可以參考的健康資料：\n{context}")
            )
        if history:
            chat_messages.append(
                HumanMessage(content=f"過去的對話摘要如下：\n{history}")
            )

        chat_messages.append(HumanMessage(content=user_input))

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
