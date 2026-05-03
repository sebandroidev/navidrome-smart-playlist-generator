import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateDB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._init()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init(self):
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS tracks (
                    id              TEXT PRIMARY KEY,
                    nav_id          TEXT,
                    beets_id        INTEGER,
                    title           TEXT NOT NULL,
                    artist          TEXT NOT NULL,
                    albumartist     TEXT,
                    album           TEXT,
                    genre           TEXT,
                    year            INTEGER,
                    format          TEXT,
                    bitrate         INTEGER,
                    play_count      INTEGER DEFAULT 0,
                    last_played     TEXT,
                    starred         INTEGER DEFAULT 0,
                    user_rating     INTEGER DEFAULT 0,
                    composite_score REAL DEFAULT 0.0,
                    audio_features  TEXT,
                    updated_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS playlist_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    playlist_type   TEXT NOT NULL,
                    generated_at    TEXT NOT NULL,
                    track_ids       TEXT NOT NULL,
                    nav_playlist_id TEXT
                );

                CREATE TABLE IF NOT EXISTS genre_clusters (
                    genre       TEXT PRIMARY KEY,
                    cluster_id  INTEGER DEFAULT 0,
                    weight      REAL DEFAULT 1.0,
                    play_count  INTEGER DEFAULT 0,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS config_overrides (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

    # ── track upsert ──────────────────────────────────────────────────────────

    def upsert_track(self, track: dict):
        with self._connect() as con:
            con.execute("""
                INSERT INTO tracks
                    (id, nav_id, beets_id, title, artist, albumartist, album, genre,
                     year, format, bitrate, play_count, last_played, starred,
                     user_rating, composite_score, audio_features, updated_at)
                VALUES
                    (:id, :nav_id, :beets_id, :title, :artist, :albumartist, :album, :genre,
                     :year, :format, :bitrate, :play_count, :last_played, :starred,
                     :user_rating, :composite_score, :audio_features, :updated_at)
                ON CONFLICT(id) DO UPDATE SET
                    nav_id          = excluded.nav_id,
                    beets_id        = excluded.beets_id,
                    title           = excluded.title,
                    artist          = excluded.artist,
                    albumartist     = excluded.albumartist,
                    album           = excluded.album,
                    genre           = excluded.genre,
                    year            = excluded.year,
                    format          = excluded.format,
                    bitrate         = excluded.bitrate,
                    play_count      = excluded.play_count,
                    last_played     = excluded.last_played,
                    starred         = excluded.starred,
                    user_rating     = excluded.user_rating,
                    composite_score = excluded.composite_score,
                    audio_features  = COALESCE(excluded.audio_features, tracks.audio_features),
                    updated_at      = excluded.updated_at
            """, {
                "id":             track["id"],
                "nav_id":         track.get("nav_id"),
                "beets_id":       track.get("beets_id"),
                "title":          track["title"],
                "artist":         track["artist"],
                "albumartist":    track.get("albumartist"),
                "album":          track.get("album"),
                "genre":          track.get("genre"),
                "year":           track.get("year"),
                "format":         track.get("format"),
                "bitrate":        track.get("bitrate"),
                "play_count":     track.get("play_count", 0),
                "last_played":    track.get("last_played"),
                "starred":        int(track.get("starred", False)),
                "user_rating":    track.get("user_rating", 0),
                "composite_score": track.get("composite_score", 0.0),
                "audio_features": json.dumps(track["audio_features"])
                                  if track.get("audio_features") else None,
                "updated_at":     _now(),
            })

    def update_score(self, track_id: str, score: float):
        with self._connect() as con:
            con.execute(
                "UPDATE tracks SET composite_score=?, updated_at=? WHERE id=?",
                (score, _now(), track_id),
            )

    def get_all_tracks(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM tracks").fetchall()
        return [self._deserialize(dict(r)) for r in rows]

    def get_track(self, track_id: str) -> Optional[dict]:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM tracks WHERE id=?", (track_id,)
            ).fetchone()
        return self._deserialize(dict(row)) if row else None

    @staticmethod
    def _deserialize(row: dict) -> dict:
        """Deserialize JSON columns back to Python objects."""
        af = row.get("audio_features")
        if isinstance(af, str):
            try:
                row["audio_features"] = json.loads(af)
            except Exception:
                row["audio_features"] = None
        return row

    def track_count(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]

    def play_coverage(self) -> float:
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            played = con.execute(
                "SELECT COUNT(*) FROM tracks WHERE play_count > 0"
            ).fetchone()[0]
        return round(played / total * 100, 1) if total else 0.0

    def top_genres(self, limit: int = 10) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("""
                SELECT genre, SUM(play_count) as total
                FROM tracks
                WHERE genre IS NOT NULL AND genre != ''
                GROUP BY genre
                ORDER BY total DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [{"genre": r["genre"], "play_count": r["total"]} for r in rows]

    def avg_score(self) -> float:
        with self._connect() as con:
            val = con.execute(
                "SELECT AVG(composite_score) FROM tracks"
            ).fetchone()[0]
        return round(val or 0.0, 4)

    # ── playlist history ───────────────────────────────────────────────────────

    def save_playlist(self, playlist_type: str, track_ids: list[str],
                      nav_playlist_id: Optional[str] = None):
        with self._connect() as con:
            con.execute("""
                INSERT INTO playlist_history (playlist_type, generated_at, track_ids, nav_playlist_id)
                VALUES (?, ?, ?, ?)
            """, (playlist_type, _now(), json.dumps(track_ids), nav_playlist_id))

    def get_recent_playlist_track_ids(self, playlist_type: str, n: int) -> set[str]:
        with self._connect() as con:
            rows = con.execute("""
                SELECT track_ids FROM playlist_history
                WHERE playlist_type=?
                ORDER BY generated_at DESC
                LIMIT ?
            """, (playlist_type, n)).fetchall()
        ids: set[str] = set()
        for r in rows:
            ids.update(json.loads(r["track_ids"]))
        return ids

    def get_playlist_history(self, playlist_type: str, limit: int = 5) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("""
                SELECT * FROM playlist_history
                WHERE playlist_type=?
                ORDER BY generated_at DESC
                LIMIT ?
            """, (playlist_type, limit)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["track_ids"] = json.loads(d["track_ids"])
            result.append(d)
        return result

    def last_run(self, playlist_type: str) -> Optional[str]:
        with self._connect() as con:
            row = con.execute("""
                SELECT generated_at FROM playlist_history
                WHERE playlist_type=?
                ORDER BY generated_at DESC
                LIMIT 1
            """, (playlist_type,)).fetchone()
        return row["generated_at"] if row else None

    # ── genre cluster weights ─────────────────────────────────────────────────

    def refresh_genre_weights(self, tracks: list[dict]):
        if not tracks:
            return
        totals: dict[str, int] = {}
        for t in tracks:
            g = (t.get("genre") or "").strip()
            if g:
                totals[g] = totals.get(g, 0) + (t.get("play_count") or 0)

        if not totals:
            return

        max_plays = max(totals.values()) or 1
        now = _now()
        with self._connect() as con:
            for genre, count in totals.items():
                weight = count / max_plays
                con.execute("""
                    INSERT INTO genre_clusters (genre, weight, play_count, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(genre) DO UPDATE SET
                        weight=excluded.weight,
                        play_count=excluded.play_count,
                        updated_at=excluded.updated_at
                """, (genre, weight, count, now))

    def get_genre_weight(self, genre: str) -> float:
        if not genre:
            return 0.5
        with self._connect() as con:
            row = con.execute(
                "SELECT weight FROM genre_clusters WHERE genre=?", (genre,)
            ).fetchone()
        return row["weight"] if row else 0.5
