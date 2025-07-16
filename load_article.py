from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from embedding import to_vector

# 連線到 Milvus
connections.connect(uri="http://localhost:19530")


# ----------讀取衛教文章 ----------
with open("article.txt", "r", encoding="utf-8") as f:
    content = f.read()

# ----------語意導向切段 ----------
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    chunk_size=200,
    chunk_overlap=50,
)
chunks = splitter.split_text(content)

# ----------向量化 ----------
chunks_vector = to_vector(chunks)


VECTOR_DIM = chunks_vector.shape[1]

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主鍵
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
]

schema = CollectionSchema(fields=fields, description="For RAG search")

assert len(chunks) == len(chunks_vector), "chunks 與 chunks_vector 數量不一致"


# ----------建 collection 並插入資料 ----------
COLLECTION_NAME = "demo1"
VECTOR_DIM = chunks_vector.shape[1]  # 假設 chunks_vector 是 numpy array

collection_name = "demo1"

if utility.has_collection(collection_name):
    Collection(collection_name).drop()

collection = Collection(name=collection_name, schema=schema)

texts = chunks  # list[str]
vectors = [vec.tolist() for vec in chunks_vector]  # list[list[float]]

collection.insert([texts, vectors])

collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "COSINE",  # 可選：COSINE / IP / L2
        "index_type": "IVF_FLAT",  # 可選：FLAT / IVF_FLAT / HNSW / ANNOY
        "params": {"nlist": 128},
    },
)

print("已將資料載入 Milvus")
