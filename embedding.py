import os
from openai import OpenAI
from typing import Union, List
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # 請設置環境變數

def to_vector(text: Union[str, List[str]], normalize: bool = True) -> List[float]:
    if isinstance(text, str):
        inputs = [text]
    elif isinstance(text, list):
        inputs = text
    else:
        raise TypeError("輸入必須為 str 或 List[str]")

    # 呼叫 OpenAI API
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=inputs
    )

    vectors = [r.embedding for r in response.data]

    # 單一輸入時回傳一維向量
    if isinstance(text, str):
        return vectors[0]
    return vectors
