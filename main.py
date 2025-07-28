import os
import json
from time import time
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import Collection, connections
from embedding import to_vector

# 載入 .env
load_dotenv()


class Bot:
    def __init__(self, user_id: str):
        """初始化 Bot 實例"""
        self.user_id = user_id
        self.chat_history = []
        
        # 初始化 LLM
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = "gpt-4o-mini"
        
        # Tool 設定
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_milvus",
                    "description": "查詢健康知識庫以輔助回答使用者的健康問題",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "使用者提出的健康問題",
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        # System Prompt
        self.system_prompt = """
你是一位會說台灣閩南語的健康陪伴機器人。你的工作是陪伴長者聊天，若問題與健康知識有關，且你不確定答案時，可以使用資料庫（search_milvus）查詢後再回答。

請保持溫暖、親切、輕鬆的語氣，回覆時盡量使用台灣閩南語，必要時穿插中文幫助理解。

如果你有用到資料庫查詢，請將查到的內容融合成自己的語氣回答，不要原文貼上。
""".strip()

    def search_milvus(self, query: str) -> str:
        """Milvus 查詢函式，只返回相似度高於閾值的結果"""
        try:
            # 獲取相似度閾值
            similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD"))
            
            connections.connect(alias="default", uri="http://localhost:19530")
            collection = Collection("copd_qa")
            collection.load()
            user_vec = to_vector(query)
            # 確保 user_vec 是正確的格式，to_vector 已返回 list
            if not isinstance(user_vec, list):
                user_vec = user_vec.tolist() if hasattr(user_vec, 'tolist') else list(user_vec)
            
            results = collection.search(
                data=[user_vec],  # user_vec 已經是 list，包在 [] 中成為向量列表
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=5,  # 增加搜索數量以便篩選
                output_fields=["question", "answer", "category"],
            )
            connections.disconnect(alias="default")
            
            relevant_chunks = []
            for hit in results[0]:
                score = hit.score
                # 只有相似度高於閾值的結果才加入
                if score >= similarity_threshold:
                    q = hit.entity.get("question")
                    a = hit.entity.get("answer")
                    cat = hit.entity.get("category")
                    relevant_chunks.append(f"[{cat}] (相似度: {score:.3f})\nQ: {q}\nA: {a}")
            
            if not relevant_chunks:
                return f"[查詢結果] 沒有找到相似度高於 {similarity_threshold} 的相關內容"
            
            return "\n\n".join(relevant_chunks)
        except Exception as e:
            return f"[Milvus 錯誤] {e}"

    def chat(self, user_input: str) -> str:
        """聊天主邏輯"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 加入歷史對話
        for pair in self.chat_history:
            messages.append({"role": "user", "content": pair["input"]})
            messages.append({"role": "assistant", "content": pair["output"]})
        
        messages.append({"role": "user", "content": user_input})

        # Step 1: 讓模型決定是否要使用 Tool
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        # Step 2: 若模型想使用 Tool
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            fn_args = json.loads(tool_call.function.arguments)
            result = self.search_milvus(fn_args["query"])

            # Step 3: 回傳 Tool 結果給模型，請它整合回答
            # 添加 assistant 的 tool call 消息
            messages.append({
                "role": "assistant",
                "content": "",  # 改為空字串而非 None
                "tool_calls": msg.tool_calls  # 使用正確的 tool_calls 格式
            })
            # 添加 tool 的回應消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

            final_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            reply = final_response.choices[0].message.content
        else:
            reply = msg.content

        # 保存對話歷史
        self.chat_history.append({"input": user_input, "output": reply})
        return reply

    def get_chat_history(self):
        """獲取聊天歷史"""
        return self.chat_history

    def clear_chat_history(self):
        """清除聊天歷史"""
        self.chat_history = []

    def get_user_id(self):
        """獲取用戶 ID"""
        return self.user_id


# === CLI 互動測試 ===
def main():
    print("👤 台語衛教聊天啟動")
    user_id = input("請輸入測試用 ID：").strip()
    
    # 創建 Bot 實例
    bot = Bot(user_id)
    
    print(f"\n✅ 用戶 {user_id} 的對話開始，輸入 exit 離開\n")
    while True:
        user_input = input("🧓 長輩：")
        if user_input.lower() in ["exit", "quit"]:
            break
        start = time()
        reply = bot.chat(user_input)
        print("👧 金孫：", reply)
        print(f"⏱️ 耗時：{time() - start:.2f} 秒\n")

if __name__ == "__main__":
    main()
