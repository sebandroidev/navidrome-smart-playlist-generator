import logging
import threading
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.models import (
    HealthResponse, StatsResponse, TriggerResponse, TriggerResult,
    PreviewResponse, PreviewTrack, HistoryEntry, ConfigPatch,
)

log = logging.getLogger(__name__)
router = APIRouter()

# These are injected at app startup
_cfg = None
_db  = None


def init(cfg, db):
    global _cfg, _db
    _cfg = cfg
    _db  = db


# ── health ─────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    import scheduler as sch
    return {
        "status": "ok",
        "last_run": {
            "daily":  _db.last_run("daily"),
            "weekly": _db.last_run("weekly"),
        },
        "next_run": {
            "daily":  sch.next_run("daily_jam"),
            "weekly": sch.next_run("weekly_jam"),
        },
        "library_size": _db.track_count(),
    }


# ── stats ──────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse)
def stats():
    total = _db.track_count()
    return {
        "total_tracks":      total,
        "scored_tracks":     total,
        "play_coverage_pct": _db.play_coverage(),
        "top_genres":        _db.top_genres(10),
        "avg_score":         _db.avg_score(),
    }


# ── trigger ────────────────────────────────────────────────────────────────

_running: dict[str, bool] = {"daily": False, "weekly": False}
_last_result: dict[str, dict] = {}


def _trigger_bg(playlist_type: str):
    from pipeline import run_pipeline
    _running[playlist_type] = True
    try:
        result = run_pipeline(playlist_type, _cfg, _db)
        _last_result[playlist_type] = result
    except Exception as exc:
        log.error("Trigger %s failed: %s", playlist_type, exc, exc_info=True)
        _last_result[playlist_type] = {"error": str(exc)}
    finally:
        _running[playlist_type] = False


@router.post("/trigger/{playlist_type}")
def trigger(playlist_type: str, background_tasks: BackgroundTasks):
    if playlist_type not in ("daily", "weekly"):
        raise HTTPException(404, "Unknown playlist type. Use 'daily' or 'weekly'.")
    if _running.get(playlist_type):
        return {"accepted": False, "playlist_type": playlist_type,
                "message": "Already running"}
    background_tasks.add_task(_trigger_bg, playlist_type)
    return {"accepted": True, "playlist_type": playlist_type,
            "message": f"{playlist_type.title()} Jam refresh started"}


@router.get("/trigger/{playlist_type}/result")
def trigger_result(playlist_type: str):
    if playlist_type not in ("daily", "weekly"):
        raise HTTPException(404, "Unknown playlist type")
    result = _last_result.get(playlist_type)
    if not result:
        return {"status": "no_result_yet"}
    return result


# ── preview ────────────────────────────────────────────────────────────────

@router.get("/playlist/{playlist_type}/preview", response_model=PreviewResponse)
def preview(playlist_type: str):
    if playlist_type not in ("daily", "weekly"):
        raise HTTPException(404, "Unknown playlist type")
    from pipeline import preview_playlist
    tracks = preview_playlist(playlist_type, _cfg, _db)
    return {
        "playlist_type": playlist_type,
        "tracks": tracks,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── history ────────────────────────────────────────────────────────────────

@router.get("/playlist/{playlist_type}/history")
def history(playlist_type: str, limit: int = 5):
    if playlist_type not in ("daily", "weekly"):
        raise HTTPException(404, "Unknown playlist type")
    rows = _db.get_playlist_history(playlist_type, limit)
    return [
        {
            "id":              r["id"],
            "playlist_type":   r["playlist_type"],
            "generated_at":    r["generated_at"],
            "track_count":     len(r["track_ids"]),
            "nav_playlist_id": r.get("nav_playlist_id"),
        }
        for r in rows
    ]


# ── config ─────────────────────────────────────────────────────────────────

@router.get("/config")
def get_config_endpoint():
    from config import get_config
    cfg = get_config()
    return {
        "scoring_weights": {
            "play_count":     cfg.scoring.weights.play_count,
            "recency":        cfg.scoring.weights.recency,
            "rating":         cfg.scoring.weights.rating,
            "genre_affinity": cfg.scoring.weights.genre_affinity,
            "discovery_bonus":cfg.scoring.weights.discovery_bonus,
        },
        "recency_halflife_days":  cfg.scoring.recency_halflife_days,
        "daily_cron":             cfg.daily.cron,
        "daily_track_count":      cfg.daily.track_count,
        "daily_comfort_ratio":    cfg.daily.comfort_ratio,
        "weekly_cron":            cfg.weekly.cron,
        "weekly_track_count":     cfg.weekly.track_count,
        "weekly_comfort_ratio":   cfg.weekly.comfort_ratio,
        "ollama_enabled":         cfg.ollama.enabled,
        "listenbrainz_enabled":   cfg.listenbrainz.enabled,
        "audio_analysis_enabled": cfg.audio_analysis.enabled,
    }


@router.patch("/config")
def patch_config_endpoint(body: ConfigPatch):
    from config import patch_config
    cfg = patch_config(body.updates)
    global _cfg
    _cfg = cfg
    return {"ok": True, "message": "Config updated"}


# ── rescan ─────────────────────────────────────────────────────────────────

@router.post("/rescan")
def rescan(background_tasks: BackgroundTasks):
    def _do():
        from pipeline import ingest_and_score
        from ingestion import cache as cache_mod
        cache_mod.clear()
        ingest_and_score(_cfg, _db)
    background_tasks.add_task(_do)
    return {"accepted": True, "message": "Rescan started"}
