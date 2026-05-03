import logging
import threading
import time
import requests
from config import AppConfig

log = logging.getLogger(__name__)

_TG_API = "https://api.telegram.org/bot{token}/{method}"


# ── outbound notifications ─────────────────────────────────────────────────

def notify(message: str, cfg: AppConfig):
    if not cfg.telegram.enabled:
        return
    if not cfg.telegram.bot_token or not cfg.telegram.chat_id:
        return
    try:
        url = _TG_API.format(token=cfg.telegram.bot_token, method="sendMessage")
        requests.post(
            url,
            json={"chat_id": cfg.telegram.chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as exc:
        log.warning("Telegram notification failed: %s", exc)


# ── inbound bot commands ───────────────────────────────────────────────────

_HELP = (
    "🎵 *Orly Jams commands*\n"
    "/daily — refresh Daily Jam\n"
    "/weekly — refresh Weekly Jam\n"
    "/genres — generate Genre Mixes\n"
    "/moods — generate Mood Mixes\n"
    "/rescan — rescan library\n"
    "/stats — library stats\n"
    "/preview\\_daily — preview Daily Jam\n"
    "/preview\\_weekly — preview Weekly Jam\n"
    "/help — show this message"
)


def _api_call(method: str, token: str, **kwargs):
    url = _TG_API.format(token=token, method=method)
    try:
        r = requests.post(url, json=kwargs, timeout=10)
        return r.json()
    except Exception as exc:
        log.debug("Telegram API %s error: %s", method, exc)
        return {}


def _send(token: str, chat_id: str, text: str):
    _api_call("sendMessage", token, chat_id=chat_id, text=text, parse_mode="Markdown")


def _handle_command(text: str, base_url: str) -> str:
    """
    Process a bot command. Makes internal HTTP calls to the local API server.
    Returns the reply text to send back.
    """
    cmd = text.strip().lstrip("/").split()[0].split("@")[0].lower()
    sess = requests.Session()

    try:
        if cmd == "daily":
            r = sess.post(f"{base_url}/trigger/daily", timeout=5)
            d = r.json()
            return "▶ Daily Jam refresh started" if d.get("accepted") else f"Already running"

        elif cmd == "weekly":
            r = sess.post(f"{base_url}/trigger/weekly", timeout=5)
            d = r.json()
            return "▶ Weekly Jam refresh started" if d.get("accepted") else "Already running"

        elif cmd == "genres":
            r = sess.post(f"{base_url}/trigger/clusters", timeout=5)
            d = r.json()
            return "▶ Genre Mixes started" if d.get("accepted") else "Already running"

        elif cmd == "moods":
            r = sess.post(f"{base_url}/trigger/moods", timeout=5)
            d = r.json()
            return "▶ Mood Mixes started" if d.get("accepted") else "Already running"

        elif cmd == "rescan":
            sess.post(f"{base_url}/rescan", timeout=5)
            return "↻ Library rescan started"

        elif cmd == "stats":
            h = sess.get(f"{base_url}/health", timeout=5).json()
            s = sess.get(f"{base_url}/stats", timeout=5).json()
            top = ", ".join(g["genre"] for g in s.get("top_genres", [])[:3])
            return (
                f"📊 *Library*\n"
                f"🎵 {h.get('library_size', '?')} tracks\n"
                f"▶ Play coverage: {s.get('play_coverage_pct', '?')}%\n"
                f"⭐ Avg score: {s.get('avg_score', '?')}\n"
                f"🎸 Top genres: {top or '—'}"
            )

        elif cmd in ("preview_daily", "preview_weekly"):
            ptype = "daily" if "daily" in cmd else "weekly"
            r = sess.get(f"{base_url}/playlist/{ptype}/preview", timeout=10).json()
            tracks = r.get("tracks", [])[:5]
            if not tracks:
                return f"No {ptype} preview available — run /rescan first"
            lines = "\n".join(
                f"{i+1}. {t.get('artist','?')} — {t.get('title','?')}"
                for i, t in enumerate(tracks)
            )
            return f"🎵 *{ptype.title()} Jam preview* (top 5)\n{lines}"

        elif cmd == "help":
            return _HELP

        else:
            return f"Unknown command: /{cmd}\n{_HELP}"

    except Exception as exc:
        log.warning("Bot command /%s error: %s", cmd, exc)
        return f"❌ Error processing /{cmd}: {exc}"


def _poll_loop(cfg: AppConfig, base_url: str):
    """Long-poll Telegram getUpdates. Runs as a daemon thread."""
    token   = cfg.telegram.bot_token
    chat_id = cfg.telegram.chat_id
    offset  = 0
    sess    = requests.Session()

    # Give the FastAPI server a moment to finish starting
    time.sleep(3)
    log.info("Telegram bot polling started")

    while True:
        try:
            url = _TG_API.format(token=token, method="getUpdates")
            r = sess.get(url, params={"timeout": 30, "offset": offset}, timeout=35)
            updates = r.json().get("result", [])
            for upd in updates:
                offset = upd["update_id"] + 1
                msg  = upd.get("message", {})
                text = msg.get("text", "")
                if not text.startswith("/"):
                    continue
                reply = _handle_command(text, base_url)
                _send(token, chat_id, reply)
        except Exception as exc:
            log.debug("Bot poll loop error: %s", exc)
            time.sleep(5)


def start_bot_polling(cfg: AppConfig, port: int = 7070):
    """Start the Telegram command bot in a background daemon thread."""
    if not cfg.telegram.enabled or not cfg.telegram.bot_token:
        return
    base_url = f"http://localhost:{port}"
    t = threading.Thread(target=_poll_loop, args=(cfg, base_url), daemon=True, name="tg-bot")
    t.start()
