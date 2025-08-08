# Filename: toolkits/redis_store.py
# -*- coding: utf-8 -*-
import os, json, time, hashlib
import redis
from functools import lru_cache
from typing import List, Tuple, Optional, Dict

REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", 86400))
ALERT_STREAM_KEY = os.getenv("ALERT_STREAM_KEY", "alerts:stream")
ALERT_STREAM_GROUP = os.getenv("ALERT_STREAM_GROUP", "case_mgr")

@lru_cache(maxsize=1)
def get_redis() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)

def _touch_ttl(keys: List[str]) -> None:
    if not keys: return
    r = get_redis(); p = r.pipeline()
    for k in keys: p.pexpire(k, REDIS_TTL_SECONDS * 1000)
    p.execute()

def ensure_active_state(user_id: str) -> None:
    r = get_redis()
    key = f"session:{user_id}:state"
    r.set(key, "ACTIVE", nx=True)
    _touch_ttl([key])

def try_register_request(user_id: str, request_id: str) -> bool:
    r = get_redis(); key = f"processed:{user_id}:{request_id}"
    return bool(r.set(key, "1", nx=True, ex=REDIS_TTL_SECONDS))

def make_request_id(user_id: str, text: str, now_ms: Optional[int] = None) -> str:
    if now_ms is None: now_ms = int(time.time()*1000)
    bucket = now_ms // 3000
    return hashlib.sha1(f"{user_id}|{text}|{bucket}".encode()).hexdigest()

# --- Conversation data ---
def append_round(user_id: str, round_obj: Dict) -> None:
    r = get_redis(); key = f"session:{user_id}:history"
    r.rpush(key, json.dumps(round_obj, ensure_ascii=False))
    ensure_active_state(user_id)
    _touch_ttl([key, f"session:{user_id}:summary:text",
                f"session:{user_id}:summary:rounds",
                f"session:{user_id}:alerts", f"session:{user_id}:state"])

def history_len(user_id: str) -> int:
    return get_redis().llen(f"session:{user_id}:history")

def fetch_unsummarized_tail(user_id: str, k: int = 6) -> List[Dict]:
    r = get_redis()
    cursor = int(r.get(f"session:{user_id}:summary:rounds") or 0)
    items = r.lrange(f"session:{user_id}:history", cursor, -1)
    return [json.loads(x) for x in items[-k:]]

def fetch_all_history(user_id: str) -> List[Dict]:
    r = get_redis(); items = r.lrange(f"session:{user_id}:history", 0, -1)
    return [json.loads(x) for x in items]

def get_summary(user_id: str) -> Tuple[str, int]:
    r = get_redis()
    text = r.get(f"session:{user_id}:summary:text") or ""
    rounds = int(r.get(f"session:{user_id}:summary:rounds") or 0)
    return text, rounds

# --- Peek 下一段 / 剩餘（不寫，容忍競態；真正原子在 commit） ---
def peek_next_n(user_id: str, n: int) -> Tuple[Optional[int], List[Dict]]:
    r = get_redis()
    cursor = int(r.get(f"session:{user_id}:summary:rounds") or 0)
    total = r.llen(f"session:{user_id}:history")
    if (total - cursor) < n:
        return None, []
    items = r.lrange(f"session:{user_id}:history", cursor, cursor + n - 1)
    return cursor, [json.loads(x) for x in items]

def peek_remaining(user_id: str) -> Tuple[int, List[Dict]]:
    r = get_redis()
    cursor = int(r.get(f"session:{user_id}:summary:rounds") or 0)
    total = r.llen(f"session:{user_id}:history")
    if total <= cursor:
        return cursor, []
    items = r.lrange(f"session:{user_id}:history", cursor, total - 1)
    return cursor, [json.loads(x) for x in items]

# --- CAS 提交分段摘要（WATCH/MULTI/EXEC） ---
def commit_summary_chunk(user_id: str, expected_cursor: int, advance: int, add_text: str) -> bool:
    r = get_redis()
    ckey = f"session:{user_id}:summary:rounds"
    tkey = f"session:{user_id}:summary:text"
    with r.pipeline() as p:
        while True:
            try:
                p.watch(ckey, tkey)
                cur = int(p.get(ckey) or 0)
                if cur != expected_cursor:
                    p.unwatch(); return False
                old = p.get(tkey) or ""
                new = (old + ("\n\n" if old else "") + (add_text or "").strip()) if add_text else old
                p.multi()
                p.set(tkey, new)
                p.set(ckey, cur + int(advance))
                p.execute()
                _touch_ttl([ckey, tkey])
                return True
            except redis.WatchError:
                # 競態衝突，重試或直接返回 False
                return False

# --- Alerts：Streams + per-user 快照 ---
def ensure_alert_group() -> None:
    r = get_redis()
    try:
        r.xgroup_create(name=ALERT_STREAM_KEY, groupname=ALERT_STREAM_GROUP, id='$', mkstream=True)
    except redis.ResponseError as e:
        if 'BUSYGROUP' not in str(e): raise

def xadd_alert(user_id: str, reason: str, severity: str = "info", extra: Optional[Dict] = None) -> str:
    ensure_alert_group()
    r = get_redis()
    fields = {"user_id": user_id, "reason": reason, "severity": severity, "ts": str(int(time.time()*1000))}
    if extra: fields["extra"] = json.dumps(extra, ensure_ascii=False)
    xid = r.xadd(ALERT_STREAM_KEY, fields)
    r.rpush(f"session:{user_id}:alerts", json.dumps(fields, ensure_ascii=False))
    _touch_ttl([f"session:{user_id}:alerts"])
    return xid

def pop_all_alerts(user_id: str) -> List[Dict]:
    r = get_redis()
    key = f"session:{user_id}:alerts"
    with r.pipeline() as p:
        p.lrange(key, 0, -1); p.delete(key)
        items, _ = p.execute()
    return [json.loads(x) for x in items]

# --- Purge 整個 user session ---
def purge_user_session(user_id: str) -> int:
    r = get_redis()
    keys = [f"session:{user_id}:history", f"session:{user_id}:summary:text",
            f"session:{user_id}:summary:rounds", f"session:{user_id}:alerts", f"session:{user_id}:state"]
    p = r.pipeline()
    for k in keys: p.delete(k)
    res = p.execute()
    return sum(res)
