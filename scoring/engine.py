import logging
from typing import Optional
from config import AppConfig
from scoring.signals import (
    play_count_signal, recency_signal, rating_signal,
    genre_affinity_signal, discovery_bonus_signal, listenbrainz_signal,
)
from state.db import StateDB

log = logging.getLogger(__name__)


def score_tracks(tracks: list[dict], cfg: AppConfig, db: StateDB) -> list[dict]:
    """
    Compute composite_score for every track in-place, persist to DB, return tracks.
    Each track dict gains: composite_score, signal_breakdown.
    """
    if not tracks:
        return tracks

    max_plays    = max(t.get("play_count") or 0 for t in tracks) or 1
    max_lb_plays = max(t.get("lb_listen_count") or 0 for t in tracks) or 1
    w        = cfg.scoring.weights
    halflife = cfg.scoring.recency_halflife_days

    # Refresh genre weights from current play distribution
    db.refresh_genre_weights(tracks)

    for t in tracks:
        play_count = t.get("play_count") or 0
        genre      = (t.get("genre") or "").strip()
        genre_w    = db.get_genre_weight(genre) if genre else 0.5

        s_play  = play_count_signal(play_count, max_plays)
        s_rec   = recency_signal(t.get("last_played"), halflife)
        s_rate  = rating_signal(t.get("user_rating") or 0)
        s_genre = genre_affinity_signal(genre_w)
        s_disc  = discovery_bonus_signal(play_count)
        s_lb    = listenbrainz_signal(t.get("lb_listen_count") or 0, max_lb_plays)

        score = (
            s_play  * w.play_count +
            s_rec   * w.recency +
            s_rate  * w.rating +
            s_genre * w.genre_affinity +
            s_disc  * w.discovery_bonus +
            s_lb    * w.lb_boost
        )

        t["composite_score"] = round(score, 6)
        t["signal_breakdown"] = {
            "play_count": round(s_play, 4),
            "recency":    round(s_rec, 4),
            "rating":     round(s_rate, 4),
            "genre":      round(s_genre, 4),
            "discovery":  round(s_disc, 4),
            "lb_boost":   round(s_lb, 4),
        }

        db.update_score(t["id"], t["composite_score"])

    log.info("Scoring: scored %d tracks", len(tracks))
    return tracks
