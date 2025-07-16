import os

from dotenv import load_dotenv

load_dotenv()
template_str1 = os.getenv("RAG_PROMPT")
template_str2 = os.getenv("FALLBACK_PROMPT")
template_str3 = os.getenv("SIMILARITY_THRESHOLD")
print(template_str1)
print(template_str2)
print(template_str3)
