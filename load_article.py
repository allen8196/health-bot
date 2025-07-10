from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding import to_vector
from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

# ----------載入設定 ----------
load_dotenv()
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "demo1"
client = MilvusClient(uri=MILVUS_URI)

# ----------讀取衛教文章 ----------
with open("qa.txt", "r", encoding="utf-8") as f:
    content = f.read()

# ----------語意導向切段 ----------
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    chunk_size=100,
    chunk_overlap=50
)
chunks = splitter.split_text(content)

# ----------向量化 ----------
chunks_vector = to_vector(chunks)
assert len(chunks) == len(chunks_vector), "chunks 與 chunks_vector 數量不一致"

# ----------建 collection 並插入資料 ----------
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

dim = chunks_vector.shape[1]
client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=dim,
    metric_type="COSINE",
    consistency_level="Strong",
    schema={"text": "str"}
)

data = [
    {"text": t, "embedding": vec.tolist()}
    for t, vec in zip(chunks, chunks_vector)
]

client.insert(collection_name=COLLECTION_NAME, data=data)
print(f"✅ 已插入 {len(data)} 筆資料到 Milvus 的 {COLLECTION_NAME} collection。")