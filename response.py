from opencc import OpenCC  # 讓機器人以繁體回答
from langchain_ollama import OllamaLLM # LLM
import time # 計算執行時間

start = time.time()

text  = "我今天覺得頭痛，應該注意什麼？"
cc = OpenCC('s2t')
llm = OllamaLLM(model="yi:6b-chat")

def ask_in_traditional(prompt):
    response = llm.invoke(prompt)
    return cc.convert(response)

print(ask_in_traditional(text))


end = time.time()
print(f"耗時：{end - start:.2f} 秒")

