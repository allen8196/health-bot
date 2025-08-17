import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from pymilvus import Collection, connections
import json

load_dotenv()


def get_postgres_connection():
    """建立並返回一個 PostgreSQL 連線"""
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DB", "senior_health"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
    }
    return psycopg2.connect(**db_config, cursor_factory=RealDictCursor)


def get_milvus_collection(collection_name: str) -> Collection:
    """連接到 Milvus 並返回指定的 Collection 實例"""
    alias = "default"
    if not connections.has_connection(alias):
        connections.connect(
            alias=alias, uri=os.getenv("MILVUS_URI", "http://localhost:19530")
        )

    collection = Collection(collection_name)
    collection.load()
    return collection

def get_user_profile(line_user_id: str) -> dict:
    """
    根據 line_user_id 從 PostgreSQL 讀取使用者畫像。
    """
    profile_data = {}
    conn = None
    try:
        conn = get_postgres_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT profile_personal_background, profile_health_status, profile_life_events 
                FROM senior_users 
                WHERE line_user_id = %s
                """,
                (line_user_id,)
            )
            profile_row = cur.fetchone()
            
            if profile_row:
                # 將 JSONB 欄位合併到一個字典中，並過濾掉 None 的值
                profile_data = {k: v for k, v in profile_row.items() if v is not None}
                print(f"✅ [Profile] 成功讀取 {line_user_id} 的使用者畫像。")
            else:
                print(f"⚠️ [Profile] 在資料庫中找不到 {line_user_id} 的使用者畫像記錄。")

    except Exception as e:
        print(f"❌ [Profile] 讀取 {line_user_id} 的使用者畫像時發生錯誤: {e}")
    finally:
        if conn:
            conn.close()
            
    return profile_data