from HealthBot.agent import create_health_companion, create_guardrail_agent, auto_save_and_summary
from crewai import Crew, Task
import json
import os
import time
import threading
from pymilvus import connections

# === 全域快取 ===
class AgentManager:
    def __init__(self):
        self.guardrail_agent = create_guardrail_agent()
        self.health_agent_cache = {}

    def get_guardrail(self):
        return self.guardrail_agent

    def get_health_agent(self, user_id: str):
        if user_id not in self.health_agent_cache:
            self.health_agent_cache[user_id] = create_health_companion(user_id)
        return self.health_agent_cache[user_id]

    def release_health_agent(self, user_id: str):
        if user_id in self.health_agent_cache:
            del self.health_agent_cache[user_id]


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


def handle_user_message(agent_manager: AgentManager, user_id: str, query: str) -> str:
    guard_agent = agent_manager.get_guardrail()
    check_task = Task(
        description=f"請判斷這句話是否危險：「{query}」\n\n若包含違法、暴力、自殺、自傷或非法內容，請回覆 'BLOCK: <原因>'；若安全請回覆 'OK'。",
        expected_output="如果內容安全回覆 'OK'，如果危險回覆 'BLOCK: <具體原因>'。",
        agent=guard_agent
    )
    guard_crew = Crew(agents=[guard_agent], tasks=[check_task], verbose=True)
    guard_result = guard_crew.kickoff().raw.strip()

    if guard_result.startswith("BLOCK:"):
        return f"🚨 系統攔截：{guard_result[6:].strip()}"

    care_agent = agent_manager.get_health_agent(user_id)
    response_task = Task(
        description=f"使用者輸入：{query}，請以關懷口吻回覆。必要時可使用工具。",
        expected_output="以台語風格提供溫暖關懷的回覆，必要時使用工具搜尋相關健康資訊或進行通報。",
        agent=care_agent
    )
    care_crew = Crew(agents=[care_agent], tasks=[response_task], verbose=False)
    response = care_crew.kickoff().raw

    log_session(user_id, query, response)
    return response


class UserSession:
    def __init__(self, user_id: str, agent_manager: AgentManager, timeout: int = 30):
        self.user_id = user_id
        self.agent_manager = agent_manager
        self.timeout = timeout
        self.last_active_time = None
        self.timer_started = False

    def update_activity(self):
        self.last_active_time = time.time()
        if not self.timer_started:
            self.timer_started = True
            self.timer_thread = threading.Thread(target=self._watchdog, daemon=True)
            self.timer_thread.start()

    def _watchdog(self):
        while True:
            time.sleep(1)
            if self.last_active_time and (time.time() - self.last_active_time > self.timeout):
                print(f"\n⏳ 使用者 {self.user_id} 閒置超過 {self.timeout} 秒，自動摘要後結束對話。")
                auto_save_and_summary(self.user_id)
                self.agent_manager.release_health_agent(self.user_id)
                os._exit(0)


def main():
    connections.connect(alias="default", uri="http://localhost:19530")
    agent_manager = AgentManager()
    session_pool = {}
    user_id = "test_user"  # 單一測試用戶，但保留衝突避免結構

    print("✅ 系統啟動，閒置 30 秒將自動總結對話並結束。")

    if user_id not in session_pool:
        session_pool[user_id] = UserSession(user_id, agent_manager)
    session = session_pool[user_id]

    try:
        while True:
            query = input("🧓 長輩：").strip()
            session.update_activity()

            reply = handle_user_message(agent_manager, user_id, query)
            print("👧 金孫：", reply)

    except KeyboardInterrupt:
        print("\n📝 中斷偵測，自動儲存摘要...")
        auto_save_and_summary(user_id)
        agent_manager.release_health_agent(user_id)


if __name__ == "__main__":
    main()