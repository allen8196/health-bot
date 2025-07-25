import json
import os

with open("intent.json", "r", encoding="utf-8") as f:
    categories = json.load(f)

    rag_keywords = "\n".join(f"- {k}" for k in categories.get("rag", []))
    print("rag_keywords", rag_keywords)
    chat_keywords = "\n".join(f"- {k}" for k in categories.get("chat", []))
    print("chat_keywords", chat_keywords)
