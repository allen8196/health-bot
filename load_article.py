from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import pandas as pd
from embedding import to_vector  # 保留你的向量化邏輯

# 連接到 Milvus
connections.connect(uri="http://localhost:19530")

# 讀取 Excel QA 表格
df = pd.read_excel("COPD_QA.xlsx")

# 整理欄位
categories = df["類別"].astype(str).tolist()
questions = df["問題（Q）"].astype(str).tolist()
answers = df["回答（A）"].astype(str).tolist()
keywords = df["關鍵詞"].fillna("").astype(str).tolist()
notes = df["注意事項 / 補充說明"].fillna("").astype(str).tolist()

# 合併 Q + A 作為語意輸入向量
combined_texts = [q + " " + a for q, a in zip(questions, answers)]
vectors = to_vector(combined_texts)
VECTOR_DIM = vectors.shape[1]

# 建立 Collection schema
collection_name = "copd_qa"
if utility.has_collection(collection_name):
    Collection(collection_name).drop()

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="notes", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
]
schema = CollectionSchema(fields=fields, description="COPD QA 資料集")
collection = Collection(name=collection_name, schema=schema)

# 插入資料
collection.insert([
    categories,
    questions,
    answers,
    keywords,
    notes,
    vectors.tolist()
])

# 建立向量索引
collection.create_index(
    field_name="embedding",
    index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
)

collection.load()
print(f"✅ 已載入 {len(questions)} 筆 QA 資料至 Milvus collection: {collection_name}")
