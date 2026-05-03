import logging
import requests
from config import AppConfig

log = logging.getLogger(__name__)


def notify(message: str, cfg: AppConfig):
    if not cfg.telegram.enabled:
        return
    if not cfg.telegram.bot_token or not cfg.telegram.chat_id:
        log.debug("Telegram not configured, skipping notification")
        return
    try:
        url = f"https://api.telegram.org/bot{cfg.telegram.bot_token}/sendMessage"
        requests.post(
            url,
            json={
                "chat_id":    cfg.telegram.chat_id,
                "text":       message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        log.info("Telegram notification sent")
    except Exception as exc:
        log.warning("Telegram notification failed: %s", exc)
