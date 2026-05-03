import math
import logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def play_count_signal(play_count: int, max_plays: int) -> float:
    """Log-normalized play frequency: 0–1."""
    if max_plays <= 0:
        return 0.0
    return math.log1p(play_count) / math.log1p(max_plays)


def recency_signal(last_played: Optional[str], halflife_days: float = 7.0) -> float:
    """Exponential decay from last play date: 0–1 (1 = played today)."""
    dt = _parse_dt(last_played)
    if dt is None:
        return 0.0
    now = datetime.now(timezone.utc)
    days_ago = (now - dt).total_seconds() / 86400.0
    return math.exp(-days_ago * math.log(2) / halflife_days)


def rating_signal(user_rating: int) -> float:
    """Map 0–5 star rating to 0–1."""
    return max(0.0, min(5, user_rating)) / 5.0


def genre_affinity_signal(genre_weight: float) -> float:
    """Pass-through of pre-computed genre weight (0–1)."""
    return max(0.0, min(1.0, genre_weight))


def discovery_bonus_signal(play_count: int) -> float:
    """Full bonus for never-played tracks, decays quickly with play count."""
    if play_count == 0:
        return 1.0
    if play_count <= 2:
        return 0.3
    return 0.0


def listenbrainz_signal(lb_listen_count: int, max_lb_plays: int) -> float:
    """Log-normalized ListenBrainz listen count: 0–1. Zero when LB is disabled."""
    if max_lb_plays <= 0 or lb_listen_count <= 0:
        return 0.0
    return math.log1p(lb_listen_count) / math.log1p(max_lb_plays)
