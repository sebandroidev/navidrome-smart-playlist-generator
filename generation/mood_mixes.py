"""
Mood-based playlist generation using BPM + energy (from beets / librosa).
Segments library into Chill / Flow / Energy buckets, pushes each to Navidrome.
Works without librosa — falls back to BPM-only and genre hinting.
"""
import logging
from generation.constraints import deduplicate, filter_unmatched, fix_consecutive_artists
from ingestion import navidrome as nav_mod

log = logging.getLogger(__name__)

_MIX_SIZE = 35
_MIN_SIZE  = 12   # skip a mood if fewer than this many qualified tracks

_BPM_CHILL_MAX  = 90
_BPM_ENERGY_MIN = 120
_ENERGY_HIGH    = 0.45

_CHILL_GENRES = {
    "ambient", "classical", "jazz", "acoustic", "soul", "blues",
    "bossa nova", "folk", "lo-fi", "chillout", "downtempo", "new age",
}
_ENERGY_GENRES = {
    "electronic", "dance", "techno", "house", "drum and bass",
    "dnb", "punk", "metal", "hardstyle", "trance", "edm", "big room",
    "drum & bass", "hardcore", "gabber",
}

_MOOD_META = {
    "chill":  ("😌", "Chill Mix"),
    "flow":   ("🌊", "Flow Mix"),
    "energy": ("⚡", "Energy Mix"),
}


def _mood_bucket(track: dict) -> str:
    bpm    = track.get("bpm") or 0
    af     = track.get("audio_features")
    energy = af.get("energy") if isinstance(af, dict) else None
    genre  = (track.get("genre") or "").strip().lower()

    if bpm > 0:
        if bpm < _BPM_CHILL_MAX:
            return "chill"
        if bpm >= _BPM_ENERGY_MIN:
            if energy is None or energy >= _ENERGY_HIGH:
                return "energy"
        return "flow"

    # No BPM — use genre hints, defaulting to flow
    for g in _CHILL_GENRES:
        if g in genre:
            return "chill"
    for g in _ENERGY_GENRES:
        if g in genre:
            return "energy"
    return "flow"


def run_mood_mixes(tracks: list[dict], cfg, db) -> list[dict]:
    """Classify, constrain, and push Chill / Flow / Energy playlists. Returns result list."""
    buckets: dict[str, list[dict]] = {"chill": [], "flow": [], "energy": []}
    for t in tracks:
        if not t.get("nav_id"):
            continue
        buckets[_mood_bucket(t)].append(t)

    log.info(
        "Mood mixes: chill=%d  flow=%d  energy=%d",
        len(buckets["chill"]), len(buckets["flow"]), len(buckets["energy"]),
    )

    client = nav_mod.NavidromeClient(
        cfg.navidrome.url, cfg.navidrome.user, cfg.navidrome.password,
    )

    results = []
    for mood, pool in buckets.items():
        if len(pool) < _MIN_SIZE:
            log.info("Mood mixes: skipping %s — only %d tracks", mood, len(pool))
            continue

        pool = sorted(pool, key=lambda t: t.get("composite_score") or 0, reverse=True)
        pool = deduplicate(pool)
        pool = filter_unmatched(pool)
        pool = fix_consecutive_artists(pool)
        selected = pool[:_MIX_SIZE]

        emoji, label = _MOOD_META[mood]
        name     = f"{emoji} {label}"
        song_ids = [t["nav_id"] for t in selected]
        pl_id    = client.push_playlist(name, song_ids)
        db.save_playlist("mood_mix", [t["id"] for t in selected], pl_id)

        results.append({
            "mood":            mood,
            "name":            name,
            "track_count":     len(selected),
            "nav_playlist_id": pl_id,
        })
        log.info("Mood mixes: %s → %d tracks (playlist %s)", name, len(selected), pl_id)

    return results
