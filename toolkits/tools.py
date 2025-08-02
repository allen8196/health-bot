from crewai.tools import BaseTool
from pymilvus import Collection, connections
from embedding import to_vector
import os
import json
from openai import OpenAI
from datetime import datetime

class SearchMilvusTool(BaseTool):
    name: str = "search_milvus"
    description: str = "在 Milvus 中搜尋 COPD 相關問答，回傳相似問題與答案"

    def _run(self, query: str) -> str:
        try:
            SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD"))
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
                output_fields=["question", "answer", "category"]
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


class SummarizeConversationTool(BaseTool):
    name: str = "summarize_conversation"
    description: str = "摘要使用者最近對話紀錄，並更新 summary 檔案"

    def _run(self, user_id: str) -> str:
        session_path = f"sessions/{user_id}.json"
        summary_path = f"sessions/{user_id}_summary.json"
        if not os.path.exists(session_path):
            return "目前無可供摘要的對話紀錄。"

        with open(session_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        text = "".join([f"第{i+1}輪:\n長輩: {h['input']}\n金孫: {h['output']}\n\n" for i, h in enumerate(history)])

        llm_api = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""
請為以下對話生成摘要，涵蓋健康問題、建議重點、情緒氛圍：\n{text}請用繁體中文回答，100-150字。
"""
        try:
            res = llm_api.chat.completions.create(
                model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
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


class AlertCaseManagerTool(BaseTool):
    name: str = "alert_case_manager"
    description: str = "當使用者處於健康或心理緊急狀況時，通報個管師並寫入記錄"

    def _run(self, reason: str) -> str:
        os.makedirs("alerts", exist_ok=True)
        log_path = "alerts/alert_log.json"
        alert_entry = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        }

        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(alert_entry)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        return f"⚠️ 已通報個管師，事由：{reason}"


class RiskKeywordCheckTool(BaseTool):
    name: str = "risk_keyword_check"
    description: str = "檢查輸入是否包含危險、違法、自殺等關鍵詞"

    def _run(self, text: str) -> str:
        DANGEROUS_KEYWORDS = ["自殺", "跳樓", "割腕", "炸彈", "殺人", "槍", "毒品", "虐待", "暴力", "性侵"]
        hit = [kw for kw in DANGEROUS_KEYWORDS if kw in text]
        if hit:
            return f"BLOCK: 偵測到關鍵字：{'、'.join(hit)}"
        return "OK"