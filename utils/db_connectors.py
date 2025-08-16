import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from pymilvus import Collection, connections

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
