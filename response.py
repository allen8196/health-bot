from opencc import OpenCC  # 讓機器人以繁體回答
from langchain_ollama import OllamaLLM # LLM
import time # 計算執行時間
from embedding import to_vector
start = time.time()

text  = "我今天覺得頭痛，應該注意什麼？"
cc = OpenCC('s2t')
llm = OllamaLLM(model="yi:6b-chat")

def ask_in_traditional(prompt):
    response = llm.invoke(prompt)
    return cc.convert(response)

print("使用者輸入:", text)
print("使用者輸入向量化:\n",to_vector(text))
print("機器人回覆:\n",ask_in_traditional(text))


end = time.time()
print(f"耗時：{end - start:.2f} 秒")

