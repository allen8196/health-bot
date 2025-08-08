from crewai.tools import BaseTool
from pymilvus import Collection, connections
from embedding import to_vector
import os, json
from openai import OpenAI
from datetime import datetime

from toolkits.redis_store import (
    commit_summary_chunk,
    xadd_alert,
)

# === Milvus ===
_milvus_loaded = False
_collection = None

class SearchMilvusTool(BaseTool):
    name: str = "search_milvus"
    description: str = "在 Milvus 中搜尋 COPD 相關問答，回傳相似問題與答案"

    def _run(self, query: str) -> str:
        global _milvus_loaded, _collection
        try:
            if not _milvus_loaded:
                connections.connect(alias="default", uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
                _collection = Collection("copd_qa"); _collection.load(); _milvus_loaded = True
            thr = float(os.getenv("SIMILARITY_THRESHOLD", 0.6))
            vec = to_vector(query)
            if not isinstance(vec, list): vec = vec.tolist() if hasattr(vec,'tolist') else list(vec)
            res = _collection.search(
                data=[vec], anns_field="embedding",
                param={"metric_type":"COSINE", "params":{"nprobe":10}}, limit=5,
                output_fields=["question","answer","category"],
            )
            out = []
            for hit in res[0]:
                if hit.score >= thr:
                    q = hit.entity.get("question"); a = hit.entity.get("answer"); cat = hit.entity.get("category")
                    out.append(f"[{cat}] (相似度: {hit.score:.3f})\nQ: {q}\nA: {a}")
            return "\n\n".join(out) if out else "[查無高相似度結果]"
        except Exception as e:
            return f"[Milvus 錯誤] {e}"

# === 分段摘要（每 5 輪）：LLM 後 CAS 提交 ===

def summarize_chunk_and_commit(user_id: str, start_round: int, history_chunk: list) -> bool:
    if not history_chunk: return True
    text = "".join([f"第{start_round+i+1}輪:\n長輩: {h['input']}\n金孫: {h['output']}\n\n" for i,h in enumerate(history_chunk)])
    prompt = f"請將下列對話做 80-120 字摘要，聚焦：健康問題、情緒、生活要點。\n\n{text}"
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        res = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role":"system","content":"你是專業的對話摘要助手。"},{"role":"user","content":prompt}],
            temperature=0.3,
        )
        body = (res.choices[0].message.content or "").strip()
        header = f"--- 第{start_round+1}至{start_round+len(history_chunk)}輪對話摘要 ---\n"
        return commit_summary_chunk(user_id, expected_cursor=start_round, advance=len(history_chunk), add_text=header+body)
    except Exception as e:
        print(f"[摘要錯誤] {e}"); return False

class AlertCaseManagerTool(BaseTool):
    name: str = "alert_case_manager"
    description: str = "通報個管師：以 Redis Streams 送出即時告警，另存 per-user 快照。"

    def _run(self, reason: str) -> str:
        try:
            xid = xadd_alert(user_id=self.runtime_context.get("user_id", "unknown"), reason=reason, severity="high")
            return f"⚠️ 已通報個管師（事件ID: {xid}），事由：{reason}"
        except Exception as e:
            return f"[Alert 送出失敗] {e}"

class RiskKeywordCheckTool(BaseTool):
    name: str = "risk_keyword_check"
    description: str = "檢查輸入是否包含危險、違法、自殺等關鍵詞"

    def _run(self, text: str) -> str:
        DANGEROUS = ["自殺","跳樓","割腕","炸彈","殺人","槍","毒品","虐待","暴力","性侵"]
        hit = [kw for kw in DANGEROUS if kw in text]
        return f"BLOCK: 偵測到關鍵字：{'、'.join(hit)}" if hit else "OK"