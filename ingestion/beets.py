import sqlite3
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _open_ro(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Beets DB not found: {db_path}")
    uri = f"file:{path}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def get_all_tracks(db_path: str) -> list[dict]:
    """Return all tracks from the beets SQLite DB."""
    try:
        con = _open_ro(db_path)
    except FileNotFoundError as exc:
        log.warning("Beets DB unavailable: %s", exc)
        return []

    try:
        rows = con.execute("""
            SELECT
                id, path, title, artist, albumartist, album,
                genres, style, year, format, bitrate,
                bpm, initial_key, length, samplerate
            FROM items
        """).fetchall()
    except Exception as exc:
        log.error("Beets query failed: %s", exc)
        con.close()
        return []

    tracks = []
    for r in rows:
        raw_path = r["path"]
        path_str = (
            raw_path.decode("utf-8", errors="replace")
            if isinstance(raw_path, bytes)
            else (raw_path or "")
        )
        genre_raw = (r["genres"] or r["style"] or "").strip()
        genre = genre_raw.split(";")[0].strip() or None

        tracks.append({
            "beets_id":   r["id"],
            "path":       path_str,
            "title":      (r["title"] or "").strip(),
            "artist":     (r["artist"] or r["albumartist"] or "").strip(),
            "albumartist":(r["albumartist"] or "").strip(),
            "album":      (r["album"] or "").strip(),
            "genre":      genre,
            "year":       r["year"] or None,
            "format":     (r["format"] or "").lower() or None,
            "bitrate":    r["bitrate"] or None,
            "bpm":        r["bpm"] or None,
            "initial_key":r["initial_key"] or None,
            "duration":   r["length"] or None,
        })

    con.close()
    log.info("Beets: loaded %d tracks", len(tracks))
    return tracks
