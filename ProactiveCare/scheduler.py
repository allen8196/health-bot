import os

import pytz
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from tasks import check_and_trigger_dynamic_care, patrol_silent_users

load_dotenv()
TAIPEI_TZ = pytz.timezone("Asia/Taipei")

redis_jobstore = RedisJobStore(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=1,
)

scheduler = BlockingScheduler(jobstores={"default": redis_jobstore}, timezone=TAIPEI_TZ)


def main():
    scheduler.add_job(
        check_and_trigger_dynamic_care,
        trigger="interval",
        minutes=10,
        id="dynamic_care_trigger",
        name="檢查 24 小時閒置使用者",
        replace_existing=True,
    )
    scheduler.add_job(
        patrol_silent_users,
        trigger=CronTrigger(day_of_week="mon", hour=9, minute=0),
        id="weekly_patrol_job",
        name="巡檢長期沉默使用者",
        replace_existing=True,
    )

    print("🚀 主動關懷排程服務已啟動...")
    scheduler.print_jobs()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("🛑 排程服務已停止。")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
