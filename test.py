import os
from dotenv import load_dotenv

load_dotenv()
template_str1 = os.getenv("SYS_PROMPT").replace("\\n", "\n")

print(template_str1)

