from crewai import Agent
from toolkits.tools import SearchMilvusTool, AlertCaseManagerTool, summarize_chunk_and_commit, ModelGuardrailTool
from toolkits.redis_store import fetch_unsummarized_tail, fetch_all_history, get_summary, peek_next_n, peek_remaining, set_state_if, purge_user_session
from openai import OpenAI
import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
try:
    # utility 於較新版本提供 has_collection 等 API
    from pymilvus import utility  # type: ignore
except Exception:  # pragma: no cover
    utility = None  # 後續以舊法回退
from embedding import safe_to_vector
import time
from typing import Dict, Any

STM_MAX_CHARS = int(os.getenv("STM_MAX_CHARS", 1800))
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", 3000))
REFINE_CHUNK_ROUNDS = int(os.getenv("REFINE_CHUNK_ROUNDS", 20))
SUMMARY_CHUNK_SIZE = int(os.getenv("SUMMARY_CHUNK_SIZE", 5))

MEM_COLLECTION = os.getenv("MEM_COLLECTION", "user_memory")
def _get_embedding_dim():
    """動態獲取 embedding 維度，避免硬編碼造成不匹配"""
    try:
        test_vec = safe_to_vector("test")
        return len(test_vec) if test_vec else 1536
    except Exception:
        return 1536

MEM_DIM = int(os.getenv("MEM_DIM", str(_get_embedding_dim())))
MEM_THRESHOLD = float(os.getenv("MEM_THRESHOLD", "0.80"))
MEM_TOPK = int(os.getenv("MEM_TOPK", "1"))

_mem_col = None

def _ensure_mem_col() -> Collection:
    global _mem_col
    if _mem_col:
        return _mem_col
    try:
        # 已有連線則沿用；否則連一次
        try:
            connections.get_connection("default")
        except Exception:
            connections.connect(alias="default", uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
        # 建表（若不存在）
        exists = False
        try:
            if utility is not None:
                exists = utility.has_collection(MEM_COLLECTION)
        except Exception:
            exists = False
        if not exists:
            try:
                # 舊版 fallback
                exists = MEM_COLLECTION in [c.name for c in connections.get_connection("default").list_collections()]
            except Exception:
                exists = False
        if not exists:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="updated_at", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=MEM_DIM),
            ]
            schema = CollectionSchema(fields, description="per-user memory (text + embedding)")
            col = Collection(name=MEM_COLLECTION, schema=schema)
            # 向量索引 + user_id 索引
            col.create_index("embedding", {"index_type":"HNSW","metric_type":"COSINE","params":{"M":16,"efConstruction":200}})
            try:
                col.create_index("user_id", {"index_type":"TRIE"})
            except Exception:
                # 某些版本不支援 TRIE，忽略即可
                pass
        _mem_col = Collection(MEM_COLLECTION)
        _mem_col.load()
        return _mem_col
    except Exception as e:
        print(f"[mem ensure error] {e}")
        return None
def _prune_user_memory(user_id: str, keep: int = 30) -> int:
    """
    保留同一 user_id 最新的 keep 筆（依 updated_at），多的刪掉。
    回傳刪除的筆數。
    """
    col = _ensure_mem_col()
    if not col:
        return 0
    try:
        # 抓出這個 user_id 的 id 與 updated_at
        rows = col.query(
            expr=f'user_id == "{user_id}"',
            output_fields=["id", "updated_at"],
            limit=10000  # 足夠大即可；資料量更大時再做分頁
        )
    except Exception:
        return 0

    if not rows or len(rows) <= keep:
        return 0

    # 依 updated_at 由舊到新
    rows.sort(key=lambda r: r.get("updated_at", 0))
    n_over = len(rows) - keep
    to_delete_ids = [r["id"] for r in rows[:n_over] if "id" in r]

    if not to_delete_ids:
        return 0

    # 主鍵欄位名就是 schema 的 name（你定義的是 "id"）
    try:
        col.delete(expr=f"id in [{','.join(map(str, to_delete_ids))}]")
    except Exception as e:
        print(f"[prune memory delete error] {e}")
        return 0
    return len(to_delete_ids)

def _append_memory(user_id: str, text: str, vec: list) -> int:
    col = _ensure_mem_col()
    if not col or not vec or not text:
        return 0
    ms = int(time.time()*1000)
    # 按 schema 順序插入（跳過 auto_id 主鍵）
    col.insert([[user_id], [ms], [text], [vec]])
    _prune_user_memory(user_id)
    return 1

def _search_memory_top1(user_id: str, qv: list, threshold: float = MEM_THRESHOLD):
    col = _ensure_mem_col()
    if not col or not qv:
        return ""
    try:
        res = col.search(
            data=[qv], anns_field="embedding",
            param={"metric_type":"COSINE","params":{"ef":64}},
            limit=max(MEM_TOPK,1),
            expr=f'user_id == "{user_id}"',
            output_fields=["text"]
        )
        if res and len(res[0])>0:
            h = res[0][0]  # 取第一筆
            if getattr(h, "score", 0.0) >= threshold:
                try:
                    text = h.entity.get("text") or ""
                    # 過濾掉空字串的記錄
                    return text if text.strip() else ""
                except Exception:
                    return ""
    except Exception as e:
        print(f"[mem search error] {e}")
    return ""

def _ensure_user_exists(user_id: str) -> None:
    """確保該 user_id 在 Collection 中存在，如果不存在則建立空記錄"""
    col = _ensure_mem_col()
    if not col:
        return
    try:
        cnt = col.query(expr=f'user_id == "{user_id}"', output_fields=["id"], limit=1)
    except Exception:
        cnt = []
    if cnt:
        return
    
    # 建立空記錄：只有 user_id 和 updated_at，text 和 embedding 為空值
    try:
        ms = int(time.time() * 1000)
        # 插入空記錄：text 為空字串，embedding 為零向量
        zero_vec = [0.0] * MEM_DIM
        col.insert([[user_id], [ms], [""], [zero_vec]])
        print(f"[mem] 為 {user_id} 建立空記錄")
    except Exception as e:
        print(f"[mem] 建立空記錄失敗: {e}")


# ---- Prompt 構建 ----

def _shrink_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars: return text
    tail = text[-max_chars:]; idx = tail.find("--- ")
    return tail[idx:] if idx != -1 else tail

def build_prompt_from_redis(user_id: str, k: int = 6, current_input: str = "") -> Dict[str, Any]:
    """
    修改此函式，使其回傳一個包含不同記憶層次的字典，而非單一字串。
    """
    summary, _ = get_summary(user_id)
    summary_text = _shrink_tail(summary, SUMMARY_MAX_CHARS) if summary else "無"
    
    rounds = fetch_unsummarized_tail(user_id, k=max(k,1))
    def render(rs): return "\n".join([f"長輩：{r['input']}\n金孫：{r['output']}" for r in rs])
    
    stm_text = render(rounds)
    # 此處的 token 限制邏輯維持不變
    while len(stm_text) > STM_MAX_CHARS and len(rounds) > 1:
        rounds = rounds[1:]; stm_text = render(rounds)
    if len(stm_text) > STM_MAX_CHARS and rounds: stm_text = stm_text[-STM_MAX_CHARS:]
    if not stm_text: stm_text = "無"

    # --- 記憶檢索 (LTM-RAG) ---
    _ensure_user_exists(user_id)
    ltm_rag_result = "無"
    if current_input:
        qv = safe_to_vector(current_input)
        if qv:
            mem_txt = _search_memory_top1(user_id, qv, threshold=MEM_THRESHOLD)
            if mem_txt and mem_txt.strip():
                ltm_rag_result = mem_txt
    
    # 回傳一個結構化的字典
    return {
        "summary_text": summary_text,
        "stm_text": stm_text,
        "ltm_rag_result": ltm_rag_result
    }

# ---- Agents ----

def create_guardrail_agent() -> Agent:
    # 關鍵字工具退場 → 全面改為 LLM 判斷
    return Agent(
        role="風險檢查員",
        goal="攔截違法/危險/自傷/需專業人士之具體指導內容",
        backstory="你是系統第一道安全防線，只輸出嚴格判斷結果。",
        tools=[ModelGuardrailTool()],
        memory=False,
        verbose=False
    )

def create_health_companion(user_id: str) -> Agent:
    return Agent(role="健康陪伴者", goal="以台語關懷長者健康與心理狀況，必要時通報", backstory="你是會講台語的金孫型陪伴機器人，回覆溫暖務實。", tools=[SearchMilvusTool(), AlertCaseManagerTool()], memory=True, verbose=False)

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
    vec = safe_to_vector(final)
    if vec:
        _append_memory(user_id, final, vec)

# ---- Finalize：補分段摘要 → Refine → Purge ----

def finalize_session(user_id: str) -> None:
    set_state_if(user_id, expect="ACTIVE", to="FINALIZING")
    start, remaining = peek_remaining(user_id)
    if remaining:
        summarize_chunk_and_commit(user_id, start_round=start, history_chunk=remaining)
    refine_summary(user_id)
    purge_user_session(user_id)
