import logging
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

log = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _cron_parts(cron: str) -> dict:
    parts = cron.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: {cron!r}")
    minute, hour, day, month, day_of_week = parts
    return dict(
        minute=minute, hour=hour, day=day,
        month=month, day_of_week=day_of_week,
    )


def start(cfg, db):
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    from pipeline import run_pipeline

    _scheduler = BackgroundScheduler(timezone="UTC", daemon=True)

    def _run_daily():
        try:
            run_pipeline("daily", cfg, db)
        except Exception as exc:
            log.error("Daily Jam scheduler error: %s", exc, exc_info=True)

    def _run_weekly():
        try:
            run_pipeline("weekly", cfg, db)
        except Exception as exc:
            log.error("Weekly Jam scheduler error: %s", exc, exc_info=True)

    _scheduler.add_job(
        _run_daily,
        CronTrigger(**_cron_parts(cfg.daily.cron), timezone="UTC"),
        id="daily_jam",
        name="Daily Jam",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    _scheduler.add_job(
        _run_weekly,
        CronTrigger(**_cron_parts(cfg.weekly.cron), timezone="UTC"),
        id="weekly_jam",
        name="Weekly Jam",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    _scheduler.start()
    log.info(
        "Scheduler started — daily: %s  weekly: %s",
        cfg.daily.cron, cfg.weekly.cron,
    )


def next_run(job_id: str) -> str | None:
    if not _scheduler:
        return None
    job = _scheduler.get_job(job_id)
    if job and job.next_run_time:
        return job.next_run_time.isoformat()
    return None


def stop():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None
