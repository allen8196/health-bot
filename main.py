from bot import build_agent, auto_save_and_summary
import json
import os

# === CLI 主程式 ===
def main():
    user_id = input("請輸入用戶 ID：").strip()
    agent = build_agent(user_id)
    session_path = f"sessions/{user_id}.json"
    os.makedirs("sessions", exist_ok=True)
    if not os.path.exists(session_path):
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    while True:
        query = input("🧓 長輩：")
        if query.lower() in ["exit", "quit"]:
            auto_save_and_summary(user_id)
            print("👋 掰掰！")
            break
        try:
            response = agent.run(query)
            print("👧 金孫：", response)
            # === 寫入對話紀錄 ===
            with open(session_path, "r+", encoding="utf-8") as f:
                history = json.load(f)
                history.append({"input": query, "output": response})
                f.seek(0)
                json.dump(history, f, ensure_ascii=False, indent=2)
                f.truncate()
        except Exception as e:
            print("⚠️ 錯誤：", e)

if __name__ == "__main__":
    main()