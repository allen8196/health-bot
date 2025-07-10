from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-large-zh")

instruction = "為這個問題或問題生成一個向量："


def to_vector(text, normalize=True):

    if isinstance(text, str):
        return model.encode(instruction + text, normalize_embeddings=normalize)
    elif isinstance(text, list):
        texts = [instruction + t for t in text]
        return model.encode(texts, normalize_embeddings=normalize)
    else:
        raise TypeError("輸入必須為 str 或 List[str]")