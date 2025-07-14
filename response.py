from langchain_ollama import OllamaLLM # LLM
import time # 計算執行時間
from embedding import to_vector
from pymilvus import connections, Collection
from langchain.prompts import PromptTemplate

start = time.time()

text  = "我感覺我有高血壓，怎麼辦？"
llm = OllamaLLM(model="adsfaaron/taide-lx-7b-chat:q5")

def ask_in_traditional(prompt):
    # 1. 連接 Milvus 並取得 collection
    connections.connect(alias="default", uri="http://localhost:19530")
    collection = Collection("demo1")
    collection.load()

    # 2. 將使用者問題向量化
    user_vec = to_vector(prompt)

    # 3. 在 Milvus 中搜尋最相似的 top 3 chunks
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }
    # Linter可能會誤報此處的型別錯誤，但執行時是正確的
    results = collection.search(
        data=[user_vec],
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["text"]
    )

    # 4. 設定相似度門檻並過濾結果
    similarity_threshold = 0.5  # 相似度門檻，可調整
    retrieved_chunks = []
    
    print("--- RAG 檢索結果與相似度 ---")
    # Linter可能會誤報此處的型別錯誤，但執行時是正確的
    for hit in results[0]:
        print(f"ID: {hit.id}, 相似度(Score): {hit.score:.4f}, Chunk: {hit.entity.get('text')}")
        if hit.score >= similarity_threshold:
            retrieved_chunks.append(hit.entity.get('text'))
    
    # 5. 根據是否有檢索結果，決定提示內容
    if retrieved_chunks:
        print("\n--- 高於門檻，使用 RAG ---")
        context = "\n\n".join(retrieved_chunks)
        template_str = """你是一位親切的「國民孫女」，你的任務是根據提供的健康參考資料，用溫暖、關懷的口氣回答長輩的健康問題。請不要提及你正在參考資料，要讓回答聽起來像是你自己的貼心建議。

參考資料：
{context}

長輩的問題：
{question}

你的回答：
"""
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template_str
        )
        final_prompt = prompt_template.format(context=context, question=prompt)
    else:
        print("\n--- 低於門檻，使用 LLM 自有知識 ---")
        template_str = """你是一位親切的「國民孫女」，你的任務是用溫暖、關懷的口氣回答長輩的健康問題。

長輩的問題：
{question}

你的回答：
"""
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=template_str
        )
        final_prompt = prompt_template.format(question=prompt)

    # 6. 呼叫 LLM 生成回覆
    response = llm.invoke(final_prompt)
    
    connections.disconnect(alias="default")
    return response

print("使用者輸入:", text)
print("機器人回覆:\n",ask_in_traditional(text))

end = time.time()
print(f"耗時：{end - start:.2f} 秒")

