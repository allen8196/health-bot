from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding import to_vector
from insert_milvus import insert_chunks_to_milvus
from pymilvus import connections, Collection


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

chunks_vector = to_vector(chunks)

