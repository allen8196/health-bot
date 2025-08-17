import os

import requests
from dotenv import load_dotenv

load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_API_URL = "https://api.line.me/v2/bot/message/push"


def send_line_message(user_id: str, message: str) -> bool:
    """發送 LINE Push Message"""
    if not LINE_CHANNEL_ACCESS_TOKEN or not message.strip():
        print(f"[LINE Push] 缺少 Token 或訊息為空，跳過發送給 {user_id}")
        return False

    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {"to": user_id, "messages": [{"type": "text", "text": message}]}

    try:
        response = requests.post(LINE_API_URL, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            print(f"✅ [LINE Push] 成功發送訊息給 {user_id}")
            return True
        else:
            print(
                f"❌ [LINE Push] 發送失敗 (HTTP {response.status_code}): {response.text}"
            )
            return False
    except Exception as e:
        print(f"❌ [LINE Push] 發送時發生錯誤: {e}")
        return False
