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
SYSTEM_PROMPT = os.getenv("SYS_PROMPT").replace("\\n", "\n")

# 明確描述知識庫範圍
KNOWLEDGE_BASE_SCOPE = (
    "知識庫僅包含與 COPD（慢性阻塞性肺病）與呼吸道保健有關的問題與答案，"
    "範圍涵蓋：疾病症狀、日常照護、運動指導、飲食建議、呼吸訓練等。"
    "不包含藥物機轉、外科治療、非呼吸相關疾病（如糖尿病、高血壓）等其他醫療主題。"
)

class HealthChatAgent:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.llm = self._init_llm()
        self.memory = self._init_memory()
        self.dialog_count = 0

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
        """
        利用 LLM 根據知識庫範圍判斷是否需要查詢知識庫（RAG）。
        回傳 True/False
        """
        judge_prompt = (
            f"{KNOWLEDGE_BASE_SCOPE}\n\n"
            f"請判斷下列用戶發問，是否『必須查詢上述知識庫才能提供正確答案』？"
            f"若需要請只回答yes，不需要請只回答no。\n\n"
            f"用戶發問：{user_input}"
        )
        resp = self.llm.invoke([SystemMessage("你是知識庫查詢意圖判斷員"), HumanMessage(judge_prompt)])
        return "yes" in resp.lower()

    def build_prompt(self, user_input, context="", history="", step_by_step=True):
        chat_messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if context:
            chat_messages.append(HumanMessage(content=f"以下是你可以參考的健康資料：\n{context}"))
        if history:
            chat_messages.append(HumanMessage(content=f"過去的對話摘要如下：\n{history}"))
        if step_by_step:
            user_input = f"請用步驟說明方式回答。{user_input}"
        chat_messages.append(HumanMessage(content=user_input))
        return chat_messages

    def chat(self, user_input):
        # === 1. 進行意圖判斷（讓LLM根據知識庫範圍判斷要不要查RAG）===
        need_rag = self.intent_detect(user_input)
        # === 2. 檢索RAG ===
        chunks = self.search_milvus(user_input) if need_rag else []
        # === 3. 讀取過去摘要 ===
        history = self.memory.load_memory_variables({})["chat_history"]
        # === 4. 組裝 prompt 並呼叫 LLM ===
        chat_messages = self.build_prompt(user_input, context="\n\n".join(chunks), history=history)
        response = self.llm.invoke(chat_messages)
        # === 5. 儲存記憶，輪次管理 ===
        self.memory.save_context({"input": user_input}, {"output": response})
        self.dialog_count += 1
        if self.dialog_count % 3 == 0:
            self.memory.clear()
            print("📝 已對話三輪，自動摘要並重置 history！")
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
