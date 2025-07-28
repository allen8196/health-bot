import os
import json
from time import time
from datetime import datetime
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
        self.conversation_count = 0  # 對話輪數計數器
        
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

    def load_summaries(self) -> dict:
        """載入現有的摘要記錄"""
        try:
            with open("summary.json", "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_summaries(self, summaries: dict):
        """保存摘要記錄到 JSON 文件"""
        try:
            with open("summary.json", "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[摘要保存錯誤] {e}")

    def generate_summary(self) -> str:
        """使用 LLM 生成對話摘要"""
        if not self.chat_history:
            return "無對話記錄"
        
        # 準備對話內容
        conversation_text = ""
        for i, pair in enumerate(self.chat_history[-9:], 1):  # 最多取最近9輪對話
            conversation_text += f"第{i}輪:\n"
            conversation_text += f"長輩: {pair['input']}\n"
            conversation_text += f"金孫: {pair['output']}\n\n"
        
        # 生成摘要的 prompt
        summary_prompt = f"""
請為以下的台語健康陪伴機器人對話生成簡潔的摘要。
摘要應該包括：
1. 主要討論的健康話題或關心事項
2. 長輩的主要需求或問題
3. 機器人提供的建議重點
4. 整體對話的溫度和氛圍

對話內容：
{conversation_text}

請用繁體中文回答，摘要控制在100-150字內。
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一個專業的對話摘要助手，擅長提取對話重點。"},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"摘要生成失敗: {e}"

    def save_conversation_summary(self, trigger_reason="定期摘要"):
        """保存對話摘要"""
        if not self.chat_history:
            return
        
        # 生成摘要
        summary = self.generate_summary()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 載入現有摘要
        summaries = self.load_summaries()
        
        # 確保用戶記錄存在
        if self.user_id not in summaries:
            summaries[self.user_id] = {
                "user_id": self.user_id,
                "summaries": []
            }
        
        # 添加新摘要
        new_summary = {
            "timestamp": timestamp,
            "conversation_count": len(self.chat_history),
            "trigger_reason": trigger_reason,
            "summary": summary
        }
        
        summaries[self.user_id]["summaries"].append(new_summary)
        
        # 保存到文件
        self.save_summaries(summaries)
        print(f"📝 對話摘要已保存 ({trigger_reason}) - {len(self.chat_history)}輪對話")

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
        self.conversation_count += 1
        
        # 檢查是否需要觸發摘要（每3輪對話）
        if self.conversation_count % 3 == 0:
            self.save_conversation_summary(f"定期摘要-第{self.conversation_count}輪")
        
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

    def get_user_summaries(self):
        """獲取當前用戶的所有摘要記錄"""
        summaries = self.load_summaries()
        return summaries.get(self.user_id, {"user_id": self.user_id, "summaries": []})

    def print_summary_history(self):
        """顯示用戶的摘要歷史"""
        user_data = self.get_user_summaries()
        summaries = user_data.get("summaries", [])
        
        if not summaries:
            print(f"📝 用戶 {self.user_id} 暫無摘要記錄")
            return
        
        print(f"\n📚 用戶 {self.user_id} 的對話摘要歷史：")
        print("=" * 50)
        
        for i, summary in enumerate(summaries, 1):
            print(f"摘要 #{i}")
            print(f"時間：{summary.get('timestamp', 'N/A')}")
            print(f"對話輪數：{summary.get('conversation_count', 'N/A')}")
            print(f"觸發原因：{summary.get('trigger_reason', 'N/A')}")
            print(f"摘要內容：{summary.get('summary', 'N/A')}")
            print("-" * 30)


# === CLI 互動測試 ===
def main():
    print("👤 台語衛教聊天啟動")
    user_id = input("請輸入測試用 ID：").strip()
    
    # 創建 Bot 實例
    bot = Bot(user_id)
    
    print(f"\n✅ 用戶 {user_id} 的對話開始，輸入 exit 離開\n")
    print("💡 特殊指令：")
    print("   📝 'summary' - 查看摘要歷史")
    print("   🔄 'save_summary' - 手動保存當前摘要")
    print("   👋 'exit' - 退出並保存最終摘要\n")
    
    try:
        while True:
            user_input = input("🧓 長輩：")
            
            # 處理特殊指令
            if user_input.lower() in ["exit", "quit"]:
                # 在退出前保存最終摘要
                if bot.chat_history:
                    print("\n📝 正在生成最終對話摘要...")
                    bot.save_conversation_summary("對話結束摘要")
                    print("✅ 摘要已保存到 summary.json")
                print("👋 再見！")
                break
            elif user_input.lower() == "summary":
                bot.print_summary_history()
                continue
            elif user_input.lower() == "save_summary":
                if bot.chat_history:
                    bot.save_conversation_summary("手動觸發摘要")
                    print("✅ 摘要已手動保存")
                else:
                    print("⚠️ 尚無對話記錄可摘要")
                continue
            
            start = time()
            reply = bot.chat(user_input)
            print("👧 金孫：", reply)
            print(f"⏱️ 耗時：{time() - start:.2f} 秒\n")
    except KeyboardInterrupt:
        # 處理 Ctrl+C 中斷
        if bot.chat_history:
            print("\n\n📝 正在生成最終對話摘要...")
            bot.save_conversation_summary("意外中斷摘要")
            print("✅ 摘要已保存到 summary.json")
        print("\n👋 對話已中斷，再見！")

if __name__ == "__main__":
    main()
