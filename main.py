from HealthBot.agent import create_health_companion, create_guardrail_agent, auto_save_and_summary
from crewai import Crew, Task
import json
import os


def log_session(user_id: str, input_text: str, output_text: str):
    path = f"sessions/{user_id}.json"
    os.makedirs("sessions", exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    history.append({"input": input_text, "output": output_text})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def main():
    user_id = input("請輸入用戶 ID：").strip()

    while True:
        query = input("🧓 長輩：")
        if query.lower() in ["exit", "quit"]:
            auto_save_and_summary(user_id)
            print("👋 掰掰！")
            break

        # === Step 1: 執行 Guardrail Agent ===
        guard_agent = create_guardrail_agent()
        check_task = Task(
            description=f"請判斷這句話是否危險：「{query}」\n\n若包含違法、暴力、自殺、自傷或非法內容，請回覆 'BLOCK: <原因>'；若安全請回覆 'OK'。",
            expected_output="如果內容安全回覆 'OK'，如果危險回覆 'BLOCK: <具體原因>'。",
            agent=guard_agent
        )
        guard_crew = Crew(agents=[guard_agent], tasks=[check_task], verbose=False)
        guard_result = guard_crew.kickoff().raw.strip()

        if guard_result.startswith("BLOCK:"):
            print(f"🚨 系統攔截：{guard_result[6:].strip()}")
            continue

        # === Step 2: 執行健康陪伴 Agent ===
        care_agent = create_health_companion(user_id)
        response_task = Task(
            description=f"使用者輸入：{query}，請以關懷口吻回覆。必要時可使用工具。",
            expected_output="以台語風格提供溫暖關懷的回覆，必要時使用工具搜尋相關健康資訊或進行通報。",
            agent=care_agent
        )
        care_crew = Crew(agents=[care_agent], tasks=[response_task], verbose=False)
        response = care_crew.kickoff().raw

        print("👧 金孫：", response)
        log_session(user_id, query, response)


if __name__ == "__main__":
    main()