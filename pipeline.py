"""
Central pipeline: ingest → merge → score → generate → push → notify.
Called by the scheduler and the /trigger REST endpoints.
"""
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional

from config import AppConfig
from state.db import StateDB
from ingestion import navidrome as nav_mod
from ingestion import beets as beets_mod
from ingestion import cache as cache_mod
from scoring.engine import score_tracks
from generation.strategies import DailyJamStrategy, WeeklyJamStrategy
from generation.naming import generate_name
import notifier

log = logging.getLogger(__name__)

_DAILY_STRATEGY  = DailyJamStrategy()
_WEEKLY_STRATEGY = WeeklyJamStrategy()

_NAV_CACHE_TTL  = 3600   # 1h
_BEETS_CACHE_TTL = 86400  # 24h


# ── normalisation helpers ─────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase ASCII, strip punctuation — used for artist+title matching."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def _track_id(artist: str, title: str) -> str:
    return f"{_norm(artist)}::{_norm(title)}"


# ── ingest ────────────────────────────────────────────────────────────────────

def _fetch_nav_songs(cfg: AppConfig) -> list[dict]:
    cached = cache_mod.get("nav_songs", _NAV_CACHE_TTL)
    if cached is not None:
        return cached
    client = nav_mod.NavidromeClient(
        cfg.navidrome.url, cfg.navidrome.user, cfg.navidrome.password
    )
    songs = client.get_all_songs()
    cache_mod.set("nav_songs", songs)
    return songs


def _fetch_beets_tracks(cfg: AppConfig) -> list[dict]:
    cached = cache_mod.get("beets_tracks", _BEETS_CACHE_TTL)
    if cached is not None:
        return cached
    tracks = beets_mod.get_all_tracks(cfg.beets.db_path)
    cache_mod.set("beets_tracks", tracks)
    return tracks


# ── merge ─────────────────────────────────────────────────────────────────────

def _merge(nav_songs: list[dict], beets_tracks: list[dict]) -> list[dict]:
    """
    Outer-join Navidrome songs with beets tracks on normalized artist+title.
    Navidrome is the source of truth for nav_id and play stats.
    Beets enriches with genre, format, bitrate, year, path.
    """
    # build beets lookup
    beets_by_key: dict[str, dict] = {}
    for bt in beets_tracks:
        key = _track_id(bt["artist"], bt["title"])
        if key:
            beets_by_key[key] = bt

    merged: list[dict] = []
    seen_ids: set[str] = set()

    # Start from Navidrome (all tracks that Navidrome knows about)
    for s in nav_songs:
        artist = (s.get("artist") or s.get("albumArtist") or "").strip()
        title  = (s.get("title") or "").strip()
        if not artist or not title:
            continue

        tid = _track_id(artist, title)
        if tid in seen_ids:
            continue
        seen_ids.add(tid)

        bt = beets_by_key.get(tid, {})

        # parse last_played from Navidrome (ISO string or epoch)
        last_played = s.get("played")  # OpenSubsonic field
        if not last_played:
            lp_epoch = s.get("lastPlayed")
            if lp_epoch:
                try:
                    last_played = datetime.fromtimestamp(
                        int(lp_epoch), tz=timezone.utc
                    ).isoformat()
                except Exception:
                    pass

        merged.append({
            "id":           tid,
            "nav_id":       str(s.get("id", "")),
            "beets_id":     bt.get("beets_id"),
            "title":        title,
            "artist":       artist,
            "albumartist":  s.get("albumArtist", artist),
            "album":        s.get("album", bt.get("album")),
            "genre":        s.get("genre") or bt.get("genre"),
            "year":         s.get("year") or bt.get("year"),
            "format":       bt.get("format") or s.get("suffix"),
            "bitrate":      bt.get("bitrate") or s.get("bitRate"),
            "path":         bt.get("path"),
            "play_count":   int(s.get("playCount") or 0),
            "last_played":  last_played,
            "starred":      bool(s.get("starred")),
            "user_rating":  int(s.get("userRating") or 0),
            "composite_score": 0.0,
            "audio_features":  None,
        })

    log.info("Merge: %d Nav songs + %d beets tracks → %d unified records",
             len(nav_songs), len(beets_tracks), len(merged))
    return merged


# ── full refresh ──────────────────────────────────────────────────────────────

def ingest_and_score(cfg: AppConfig, db: StateDB) -> list[dict]:
    """Ingest all sources, merge, score, persist. Returns scored track list."""
    cache_mod.invalidate("nav_songs")  # always fresh on explicit call

    nav_songs = _fetch_nav_songs(cfg)
    beets_tracks = _fetch_beets_tracks(cfg)

    tracks = _merge(nav_songs, beets_tracks)

    if cfg.audio_analysis.enabled:
        from scoring.audio import analyze_tracks
        tracks = analyze_tracks(tracks, cfg.audio_analysis.cache_forever)

    score_tracks(tracks, cfg, db)

    for t in tracks:
        db.upsert_track(t)

    return tracks


# ── playlist generation ───────────────────────────────────────────────────────

def run_pipeline(playlist_type: str, cfg: AppConfig, db: StateDB) -> dict:
    """
    Full end-to-end for one playlist type ('daily' or 'weekly').
    Returns { playlist_type, track_count, nav_playlist_id, name, duration_ms }.
    """
    import time
    t0 = time.monotonic()

    tracks = ingest_and_score(cfg, db)

    if playlist_type == "daily":
        strategy = _DAILY_STRATEGY
    elif playlist_type == "weekly":
        strategy = _WEEKLY_STRATEGY
    else:
        raise ValueError(f"Unknown playlist_type: {playlist_type!r}")

    selected = strategy.select(tracks, cfg, db)

    if not selected:
        log.warning("Pipeline: no tracks selected for %s", playlist_type)
        return {"playlist_type": playlist_type, "track_count": 0,
                "nav_playlist_id": None, "name": None}

    # Optional Ollama / rule-based naming
    playlist_name = cfg.playlist_names.get(playlist_type, f"🎵 {playlist_type.title()} Jam")
    # Dynamic sub-title (not replacing the configured base name, stored separately)
    dynamic_name = generate_name(playlist_type, selected, cfg)

    # Push to Navidrome
    song_ids = [t["nav_id"] for t in selected if t.get("nav_id")]
    client = nav_mod.NavidromeClient(
        cfg.navidrome.url, cfg.navidrome.user, cfg.navidrome.password
    )
    pl_id = client.push_playlist(playlist_name, song_ids)

    # Persist history
    db.save_playlist(playlist_type, [t["id"] for t in selected], pl_id)

    duration_ms = int((time.monotonic() - t0) * 1000)
    log.info("Pipeline: %s done in %dms — %d tracks → playlist %s",
             playlist_type, duration_ms, len(selected), pl_id)

    # Telegram notification
    notifier.notify(
        f"{playlist_name} refreshed — {len(selected)} tracks ready\n"
        f"_{dynamic_name}_",
        cfg,
    )

    return {
        "playlist_type":    playlist_type,
        "track_count":      len(selected),
        "nav_playlist_id":  pl_id,
        "name":             playlist_name,
        "dynamic_name":     dynamic_name,
        "duration_ms":      duration_ms,
    }


def preview_playlist(playlist_type: str, cfg: AppConfig, db: StateDB) -> list[dict]:
    """Generate a playlist without pushing to Navidrome."""
    tracks = db.get_all_tracks()

    if not tracks:
        tracks = ingest_and_score(cfg, db)

    if playlist_type == "daily":
        selected = _DAILY_STRATEGY.select(tracks, cfg, db)
    else:
        selected = _WEEKLY_STRATEGY.select(tracks, cfg, db)

    return [
        {
            "title":           t.get("title"),
            "artist":          t.get("artist"),
            "album":           t.get("album"),
            "genre":           t.get("genre"),
            "score":           t.get("composite_score"),
            "play_count":      t.get("play_count"),
            "last_played":     t.get("last_played"),
        }
        for t in selected
    ]
