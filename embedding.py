# embed_bge_chunk.py

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

# ---------- Step 1：載入中文 Embedding 模型 ----------
print("🔄 載入 embedding 模型 BAAI/bge-small-zh 中...")
model = SentenceTransformer("BAAI/bge-small-zh")
instruction = "為這個句子生成表示以用於檢索相關文件："

# ---------- Step 2：讀取衛教文章 ----------
with open("qa.txt", "r", encoding="utf-8") as f:
    content = f.read()

# ---------- Step 3：語意導向切段 ----------
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_text(content)

# ---------- Step 4：將每個 chunk 做 embedding ----------
print(f"📄 總共有 {len(chunks)} 個段落，開始進行向量轉換...")
chunk_vectors = model.encode([instruction + chunk for chunk in chunks])

print(f"\n✅ 完成！每段向量維度：{chunk_vectors.shape[1]}")
print("📌 第一段原文：", chunks[0])
print("📌 第一段向量（前 5 維）：", chunk_vectors[0][:5])

# ---------- Step 5：使用者輸入並轉換為向量 ----------
user_input = input("\n請輸入你想詢問的內容：")
user_vector = model.encode([instruction + user_input])[0]
print("🧠 使用者輸入向量（前 5 維）：", user_vector[:5])
