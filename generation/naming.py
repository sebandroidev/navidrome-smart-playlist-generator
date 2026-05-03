import logging
import requests
from datetime import datetime, timezone
from config import AppConfig

log = logging.getLogger(__name__)

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_TIME_SLOTS = [
    (5,  12, "morning"),
    (12, 17, "afternoon"),
    (17, 21, "evening"),
    (21, 24, "night"),
    (0,   5, "late night"),
]


def _time_slot() -> str:
    h = datetime.now(timezone.utc).hour
    for start, end, label in _TIME_SLOTS:
        if start <= h < end:
            return label
    return "night"


def _day_str() -> str:
    return _DAY_NAMES[datetime.now(timezone.utc).weekday()]


def rule_based_name(playlist_type: str, tracks: list[dict]) -> str:
    slot = _time_slot()
    day  = _day_str().lower()
    genres = []
    seen: set[str] = set()
    for t in tracks[:10]:
        g = (t.get("genre") or "").strip().lower()
        if g and g not in seen:
            genres.append(g)
            seen.add(g)
    top_genres = " · ".join(genres[:2]) if genres else "mixed"

    if playlist_type == "daily":
        return f"🎵 Daily Jam · {top_genres} · {day} {slot}"
    return f"🗓 Weekly Jam · {top_genres} · week {datetime.now(timezone.utc).isocalendar()[1]}"


def ollama_name(playlist_type: str, tracks: list[dict], cfg: AppConfig) -> str:
    try:
        sample = [
            f"{t.get('artist','?')} — {t.get('title','?')} ({t.get('genre') or 'unknown'})"
            for t in tracks[:12]
        ]
        prompt = (
            f"Generate a short, creative playlist name (max 8 words, lowercase, "
            f"no quotes) for a {playlist_type} music playlist containing these tracks:\n"
            + "\n".join(sample)
        )
        resp = requests.post(
            f"{cfg.ollama.host}/api/generate",
            json={"model": cfg.ollama.model, "prompt": prompt, "stream": False},
            timeout=20,
        )
        resp.raise_for_status()
        name = resp.json().get("response", "").strip().strip('"').strip("'")
        if name:
            prefix = "🎵" if playlist_type == "daily" else "🗓"
            return f"{prefix} {name}"
    except Exception as exc:
        log.warning("Ollama naming failed: %s", exc)
    return rule_based_name(playlist_type, tracks)


def generate_name(playlist_type: str, tracks: list[dict], cfg: AppConfig) -> str:
    if cfg.ollama.enabled:
        return ollama_name(playlist_type, tracks, cfg)
    return rule_based_name(playlist_type, tracks)
