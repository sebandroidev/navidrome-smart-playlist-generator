import logging
import random
from datetime import datetime, timezone
from config import AppConfig
from state.db import StateDB
from generation.constraints import apply_all

log = logging.getLogger(__name__)


def _hours_since(last_played: str | None) -> float:
    if not last_played:
        return float("inf")
    try:
        dt = datetime.fromisoformat(last_played.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except Exception:
        return float("inf")


def _build_similarity_discovery(
    comfort_picks: list[dict],
    all_tracks: list[dict],
    exclude_ids: set[str],
    n: int,
) -> list[dict]:
    """
    Use cosine similarity to find unplayed tracks most similar to
    the comfort picks. Falls back to genre-diverse random if numpy unavailable.
    """
    try:
        from scoring.similarity import build_feature_matrix, top_k_similar
        matrix, track_ids, _ = build_feature_matrix(all_tracks)
        seed_ids = [t["id"] for t in comfort_picks if t.get("id")]
        results = top_k_similar(
            seed_ids, all_tracks, matrix, track_ids,
            k=n * 3,  # fetch extra, constraints will trim
            exclude_ids=exclude_ids | {t["id"] for t in comfort_picks},
        )
        return results[:n]
    except Exception as exc:
        log.debug("Similarity discovery unavailable (%s), falling back", exc)
        return _pick_genre_diverse(
            [t for t in all_tracks
             if t.get("nav_id") and t.get("id") not in exclude_ids
             and t not in comfort_picks],
            n,
        )


def _split_comfort(
    tracks: list[dict],
    comfort_ratio: float,
    exclude_ids: set[str],
    exclude_played_within_hours: float = 0,
) -> tuple[list[dict], list[dict]]:
    eligible = [t for t in tracks if t.get("nav_id")]
    if exclude_played_within_hours > 0:
        eligible = [
            t for t in eligible
            if _hours_since(t.get("last_played")) >= exclude_played_within_hours
        ]
    if exclude_ids:
        eligible = [t for t in eligible if t.get("id") not in exclude_ids]

    sorted_by_score = sorted(eligible, key=lambda t: t.get("composite_score", 0), reverse=True)
    n_comfort = max(round(len(sorted_by_score) * comfort_ratio), 1)
    comfort_pool = sorted_by_score[:n_comfort]
    remaining = [t for t in eligible if t not in comfort_pool]
    return comfort_pool, remaining


def _pick_genre_diverse(pool: list[dict], n: int) -> list[dict]:
    if not pool or n <= 0:
        return []
    if len(pool) <= n:
        return list(pool)
    by_genre: dict[str, list[dict]] = {}
    for t in pool:
        g = (t.get("genre") or "unknown").lower()
        by_genre.setdefault(g, []).append(t)
    selected: list[dict] = []
    genre_keys = list(by_genre.keys())
    random.shuffle(genre_keys)
    indices = {g: 0 for g in genre_keys}
    while len(selected) < n:
        added_any = False
        for g in genre_keys:
            if len(selected) >= n:
                break
            if indices[g] < len(by_genre[g]):
                selected.append(by_genre[g][indices[g]])
                indices[g] += 1
                added_any = True
        if not added_any:
            break
    return selected


class DailyJamStrategy:
    def select(self, tracks: list[dict], cfg: AppConfig, db: StateDB) -> list[dict]:
        n = cfg.daily.track_count
        comfort_ratio = cfg.daily.comfort_ratio
        exclude_hours = cfg.daily.exclude_played_within_hours

        comfort_pool, _ = _split_comfort(
            tracks, comfort_ratio,
            exclude_ids=set(),
            exclude_played_within_hours=exclude_hours,
        )
        n_comfort   = round(n * comfort_ratio)
        n_discovery = n - n_comfort
        comfort_picks = comfort_pool[:n_comfort]

        # P2: similarity-boosted discovery
        all_exclude = {t["id"] for t in comfort_picks if t.get("id")}
        discovery_picks = _build_similarity_discovery(
            comfort_picks, tracks, all_exclude, n_discovery
        )

        combined = comfort_picks + discovery_picks
        random.shuffle(combined)
        result = apply_all(combined)[:n]
        log.info("Daily: %d tracks (%d comfort, %d similarity-discovery)",
                 len(result), len(comfort_picks), len(discovery_picks))
        return result


class WeeklyJamStrategy:
    def select(self, tracks: list[dict], cfg: AppConfig, db: StateDB) -> list[dict]:
        n = cfg.weekly.track_count
        comfort_ratio = cfg.weekly.comfort_ratio
        exclude_n = cfg.weekly.exclude_last_n_weekly_playlists
        exclude_ids = db.get_recent_playlist_track_ids("weekly", exclude_n)

        comfort_pool, _ = _split_comfort(
            tracks, comfort_ratio,
            exclude_ids=exclude_ids,
            exclude_played_within_hours=0,
        )
        n_comfort   = round(n * comfort_ratio)
        n_discovery = n - n_comfort
        comfort_picks = comfort_pool[:n_comfort]

        all_exclude = exclude_ids | {t["id"] for t in comfort_picks if t.get("id")}
        discovery_picks = _build_similarity_discovery(
            comfort_picks, tracks, all_exclude, n_discovery
        )

        combined = comfort_picks + discovery_picks
        random.shuffle(combined)
        result = apply_all(combined)[:n]
        log.info("Weekly: %d tracks (%d comfort, %d similarity-discovery, excl. %d prior)",
                 len(result), len(comfort_picks), len(discovery_picks), len(exclude_ids))
        return result
