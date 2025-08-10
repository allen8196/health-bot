import os
import json
import time
import argparse
from datetime import datetime

import redis
from pymilvus import connections, Collection
# ---------- Helpers ----------

def ts2str(ms: int) -> str:
    try:
        return datetime.fromtimestamp(int(ms)/1000).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(ms)

# ---------- Milvus: read refined summaries in user_memory ----------

def load_milvus_collection(name: str) -> Collection:
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    # 若已連線會沿用
    try:
        connections.get_connection("default")
    except Exception:
        connections.connect(alias="default", uri=uri)
    col = Collection(name)
    try:
        col.load()
    except Exception:
        pass
    return col


def list_user_memory(user_id: str, collection: str, limit: int = 20):
    col = load_milvus_collection(collection)
    try:
        rows = col.query(
            expr=f'user_id == "{user_id}"',
            output_fields=["id", "user_id", "updated_at", "text"],
            limit=10000,
        )
    except Exception as e:
        print(f"[Milvus] 查詢失敗: {e}")
        return []
    # 過濾空白 text（_ensure_user_exists 可能插入空記錄）
    rows = [r for r in rows if str(r.get("text", "")).strip()]
    # 依 updated_at 由新到舊
    rows.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
    return rows[:max(1, limit)]

# ---------- Redis: read per-user snapshot list & stream ----------

def get_redis_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)


def list_user_alerts(user_id: str):
    r = get_redis_client()
    key = f"session:{user_id}:alerts"
    try:
        items = r.lrange(key, 0, -1)
    except Exception as e:
        print(f"[Redis] 讀取 {key} 失敗: {e}")
        return []
    out = []
    for it in items:
        try:
            out.append(json.loads(it))
        except Exception:
            out.append({"raw": it})
    return out


def list_stream_alerts(limit: int = 10):
    r = get_redis_client()
    stream_key = os.getenv("ALERT_STREAM_KEY", "alerts:stream")
    try:
        entries = r.xrevrange(stream_key, count=limit)
    except Exception as e:
        print(f"[Redis] 讀取 stream {stream_key} 失敗: {e}")
        return []
    out = []
    for sid, fields in entries:
        item = dict(fields)
        item["_id"] = sid
        out.append(item)
    return out

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Inspect Milvus user_memory and Redis alerts")
    parser.add_argument("--user-id", default=os.getenv("TEST_USER_ID", "test_user"))
    parser.add_argument("--mem-collection", default=os.getenv("MEM_COLLECTION", "user_memory"))
    parser.add_argument("--limit", type=int, default=10, help="最多顯示幾筆精緻摘要")
    parser.add_argument("--stream-limit", type=int, default=10, help="最多顯示幾筆 alerts:stream")
    parser.add_argument("--raw", action="store_true", help="輸出原始 JSON")
    args = parser.parse_args()

    print("\n====== Milvus: 精緻摘要（user_memory）======")
    mem_rows = list_user_memory(args.user_id, args.mem_collection, args.limit)
    if not mem_rows:
        print("(無資料)")
    else:
        for i, r in enumerate(mem_rows, 1):
            ts = ts2str(r.get("updated_at", 0))
            txt = r.get("text", "").strip()
            if args.raw:
                print(json.dumps(r, ensure_ascii=False))
            else:
                print(f"{i:>2}. @{ts}  id={r.get('id')}\n    {txt[:300]}{'…' if len(txt)>300 else ''}")

    print("\n====== Redis: per-user 快照（session:<uid>:alerts）======")
    snaps = list_user_alerts(args.user_id)
    if not snaps:
        print("(無快照資料)")
    else:
        for i, a in enumerate(snaps[-args.limit:], 1):
            if args.raw:
                print(json.dumps(a, ensure_ascii=False))
            else:
                print(f"{i:>2}. [{a.get('severity','?')}] @{ts2str(a.get('ts','0'))}  {a.get('reason','')}  (user={a.get('user_id','')})")

    print("\n====== Redis: alerts:stream（最新→較舊）======")
    events = list_stream_alerts(args.stream_limit)
    if not events:
        print("(無 stream 事件)")
    else:
        for i, e in enumerate(events, 1):
            if args.raw:
                print(json.dumps(e, ensure_ascii=False))
            else:
                print(f"{i:>2}. id={e.get('_id')} [{e.get('severity','?')}] @{ts2str(e.get('ts','0'))} user={e.get('user_id','')} | {e.get('reason','')}")

if __name__ == "__main__":
    main()