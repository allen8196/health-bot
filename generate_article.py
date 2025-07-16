import os

from langchain_ollama import OllamaLLM

# ✅ 初始化 Yi-6B 模型
llm = OllamaLLM(model="yi:6b-chat")

# ✅ 衛教主題列表
topics = [
    "高血壓的日常管理與飲食建議",
    "糖尿病的症狀與預防方法",
    "失眠的常見原因與改善策略",
    "關節炎的保健運動與飲食指引",
    "老人常見跌倒風險與預防措施",
    "便秘與腸道健康的改善方法",
    "中風後的復健與照護建議",
    "高血脂與動脈硬化的認識",
    "眼睛乾澀與視力保健技巧",
    "骨質疏鬆的預防與日常保健",
]

# ✅ 撰寫整篇內容
full_text = ""
for i, topic in enumerate(topics, start=1):
    prompt = f"""你是一位專業的健康教育專家，請撰寫一段針對長者的衛教文章，主題為「{topic}」。文章須簡明、友善、有鼓勵性，繁體中文，字數約 300～500 字。"""
    result = llm.invoke(prompt)
    full_text += f"【主題{i}：{topic}】\n{result.strip()}\n\n"

# ✅ 儲存為 txt
output_path = "article.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"✅ 已產出：{output_path}")
