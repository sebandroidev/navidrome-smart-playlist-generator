"""
Integration tests for StateDB using a real SQLite file in tmp_path.

Every public method is exercised against an actual database so that SQL
correctness, COALESCE logic, JSON serialisation round-trips, and aggregate
queries are all verified without mocks.
"""
import json
import pytest
from pathlib import Path

from state.db import StateDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_track(**kw) -> dict:
    """Return a fully-populated track dict with sensible defaults."""
    defaults = {
        "id":              "artistx::trackx",
        "nav_id":          "nav123",
        "beets_id":        1,
        "title":           "Track X",
        "artist":          "Artist X",
        "albumartist":     "Artist X",
        "album":           "Album X",
        "genre":           "hip-hop",
        "year":            2020,
        "format":          "flac",
        "bitrate":         320,
        "play_count":      5,
        "last_played":     "2026-01-01T00:00:00+00:00",
        "starred":         False,
        "user_rating":     3,
        "composite_score": 0.42,
        "audio_features":  {"bpm": 120.0, "energy": 0.7},
        "lb_listen_count": 0,
    }
    return {**defaults, **kw}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Fresh StateDB backed by a temporary SQLite file."""
    return StateDB(str(tmp_path / "test.db"))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_tables_created(self, tmp_path):
        path = str(tmp_path / "init_test.db")
        db = StateDB(path)
        # File must exist
        assert Path(path).exists()
        # All four tables must be present
        import sqlite3
        con = sqlite3.connect(path)
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        con.close()
        assert {"tracks", "playlist_history", "genre_clusters", "config_overrides"} <= tables


# ---------------------------------------------------------------------------
# Track upsert / get
# ---------------------------------------------------------------------------

class TestUpsertAndGet:
    def test_upsert_and_get_track(self, db):
        t = make_track()
        db.upsert_track(t)
        result = db.get_track(t["id"])
        assert result is not None
        assert result["id"]            == t["id"]
        assert result["nav_id"]        == t["nav_id"]
        assert result["beets_id"]      == t["beets_id"]
        assert result["title"]         == t["title"]
        assert result["artist"]        == t["artist"]
        assert result["albumartist"]   == t["albumartist"]
        assert result["album"]         == t["album"]
        assert result["genre"]         == t["genre"]
        assert result["year"]          == t["year"]
        assert result["format"]        == t["format"]
        assert result["bitrate"]       == t["bitrate"]
        assert result["play_count"]    == t["play_count"]
        assert result["last_played"]   == t["last_played"]
        assert result["user_rating"]   == t["user_rating"]
        assert abs(result["composite_score"] - t["composite_score"]) < 1e-9
        # audio_features must come back as a dict, not a string
        assert isinstance(result["audio_features"], dict)
        assert result["audio_features"] == t["audio_features"]

    def test_audio_features_serialization(self, db):
        features = {"bpm": 98.5, "energy": 0.42, "valence": 0.8}
        t = make_track(audio_features=features)
        db.upsert_track(t)
        result = db.get_track(t["id"])
        assert isinstance(result["audio_features"], dict)
        assert result["audio_features"] == features

    def test_audio_features_none_stored_as_null(self, db):
        t = make_track(audio_features=None)
        db.upsert_track(t)
        result = db.get_track(t["id"])
        assert result["audio_features"] is None

    def test_get_track_missing_returns_none(self, db):
        assert db.get_track("does::notexist") is None

    def test_upsert_idempotent(self, db):
        t = make_track()
        db.upsert_track(t)
        db.upsert_track(t)
        assert db.track_count() == 1

    def test_upsert_overwrites_scalar_fields(self, db):
        t = make_track(play_count=1, user_rating=1)
        db.upsert_track(t)
        updated = make_track(play_count=10, user_rating=5)
        db.upsert_track(updated)
        result = db.get_track(t["id"])
        assert result["play_count"]  == 10
        assert result["user_rating"] == 5

    def test_upsert_coalesce_audio_features(self, db):
        """Second upsert without audio_features must NOT overwrite existing value."""
        t = make_track(audio_features={"bpm": 120.0, "energy": 0.7})
        db.upsert_track(t)
        # Second upsert: audio_features=None → COALESCE should keep original
        t2 = make_track(audio_features=None, play_count=99)
        db.upsert_track(t2)
        result = db.get_track(t["id"])
        assert result["audio_features"] == {"bpm": 120.0, "energy": 0.7}
        assert result["play_count"] == 99  # other fields still updated


# ---------------------------------------------------------------------------
# update_score
# ---------------------------------------------------------------------------

class TestUpdateScore:
    def test_update_score(self, db):
        t = make_track(composite_score=0.1)
        db.upsert_track(t)
        db.update_score(t["id"], 0.99)
        result = db.get_track(t["id"])
        assert abs(result["composite_score"] - 0.99) < 1e-9

    def test_update_score_nonexistent_id_is_noop(self, db):
        # Should not raise; just affects 0 rows
        db.update_score("no::such", 0.5)


# ---------------------------------------------------------------------------
# track_count
# ---------------------------------------------------------------------------

class TestTrackCount:
    def test_track_count_empty(self, db):
        assert db.track_count() == 0

    def test_track_count(self, db):
        for i in range(4):
            db.upsert_track(make_track(id=f"artist{i}::track{i}"))
        assert db.track_count() == 4


# ---------------------------------------------------------------------------
# play_coverage
# ---------------------------------------------------------------------------

class TestPlayCoverage:
    def test_play_coverage_empty_db(self, db):
        assert db.play_coverage() == 0.0

    def test_play_coverage_all_unplayed(self, db):
        for i in range(3):
            db.upsert_track(make_track(id=f"a{i}::t{i}", play_count=0))
        assert db.play_coverage() == 0.0

    def test_play_coverage_all_played(self, db):
        for i in range(3):
            db.upsert_track(make_track(id=f"a{i}::t{i}", play_count=i + 1))
        assert db.play_coverage() == 100.0

    def test_play_coverage_partial(self, db):
        # 2 played out of 4 → 50 %
        for i in range(4):
            db.upsert_track(make_track(
                id=f"a{i}::t{i}",
                play_count=5 if i < 2 else 0,
            ))
        assert db.play_coverage() == 50.0


# ---------------------------------------------------------------------------
# top_genres
# ---------------------------------------------------------------------------

class TestTopGenres:
    def test_top_genres_empty(self, db):
        assert db.top_genres() == []

    def test_top_genres_sorted_by_play_count(self, db):
        db.upsert_track(make_track(id="a::rock",    genre="rock",    play_count=10))
        db.upsert_track(make_track(id="a::jazz",    genre="jazz",    play_count=3))
        db.upsert_track(make_track(id="a::hiphop",  genre="hip-hop", play_count=20))
        result = db.top_genres(limit=10)
        genres = [r["genre"] for r in result]
        counts = [r["play_count"] for r in result]
        assert genres == ["hip-hop", "rock", "jazz"]
        assert counts == [20, 10, 3]

    def test_top_genres_limit(self, db):
        for i in range(6):
            db.upsert_track(make_track(id=f"a::g{i}", genre=f"genre{i}", play_count=i + 1))
        result = db.top_genres(limit=3)
        assert len(result) == 3

    def test_top_genres_ignores_null_and_empty(self, db):
        db.upsert_track(make_track(id="a::none",  genre=None,  play_count=100))
        db.upsert_track(make_track(id="a::empty", genre="",    play_count=100))
        db.upsert_track(make_track(id="a::real",  genre="jazz", play_count=1))
        result = db.top_genres()
        assert len(result) == 1
        assert result[0]["genre"] == "jazz"

    def test_top_genres_aggregates_same_genre(self, db):
        db.upsert_track(make_track(id="a1::t1", genre="rock", play_count=7))
        db.upsert_track(make_track(id="a2::t2", genre="rock", play_count=3))
        result = db.top_genres()
        assert len(result) == 1
        assert result[0]["play_count"] == 10


# ---------------------------------------------------------------------------
# avg_score
# ---------------------------------------------------------------------------

class TestAvgScore:
    def test_avg_score_empty(self, db):
        assert db.avg_score() == 0.0

    def test_avg_score_multiple(self, db):
        scores = [0.2, 0.4, 0.6]
        for i, s in enumerate(scores):
            db.upsert_track(make_track(id=f"a{i}::t{i}", composite_score=s))
        expected = round(sum(scores) / len(scores), 4)
        assert db.avg_score() == expected


# ---------------------------------------------------------------------------
# playlist history
# ---------------------------------------------------------------------------

class TestPlaylistHistory:
    def test_save_and_get_playlist_history(self, db):
        db.save_playlist("daily", ["id1", "id2", "id3"], nav_playlist_id="pl-abc")
        db.save_playlist("daily", ["id4", "id5"],         nav_playlist_id="pl-def")
        rows = db.get_playlist_history("daily", limit=5)
        assert len(rows) == 2
        # Most recent first
        assert rows[0]["track_ids"] == ["id4", "id5"]
        assert rows[1]["track_ids"] == ["id1", "id2", "id3"]
        # track_ids must be a list, not a raw JSON string
        assert isinstance(rows[0]["track_ids"], list)

    def test_history_respects_limit(self, db):
        for i in range(5):
            db.save_playlist("weekly", [f"t{i}"])
        rows = db.get_playlist_history("weekly", limit=3)
        assert len(rows) == 3

    def test_history_is_type_filtered(self, db):
        db.save_playlist("daily",  ["d1"])
        db.save_playlist("weekly", ["w1"])
        assert len(db.get_playlist_history("daily",  limit=10)) == 1
        assert len(db.get_playlist_history("weekly", limit=10)) == 1
        assert len(db.get_playlist_history("daily",  limit=10)) == 1

    def test_history_empty_when_no_entries(self, db):
        assert db.get_playlist_history("daily") == []

    def test_nav_playlist_id_stored(self, db):
        db.save_playlist("daily", ["t1"], nav_playlist_id="nav-999")
        row = db.get_playlist_history("daily")[0]
        assert row["nav_playlist_id"] == "nav-999"


# ---------------------------------------------------------------------------
# last_run
# ---------------------------------------------------------------------------

class TestLastRun:
    def test_last_run_no_history(self, db):
        assert db.last_run("daily") is None

    def test_last_run_returns_timestamp(self, db):
        db.save_playlist("daily", ["t1"])
        ts = db.last_run("daily")
        assert ts is not None
        assert isinstance(ts, str)
        # Must be parseable as ISO 8601
        from datetime import datetime
        datetime.fromisoformat(ts)

    def test_last_run_returns_most_recent(self, db):
        db.save_playlist("daily", ["t1"])
        db.save_playlist("daily", ["t2"])
        rows = db.get_playlist_history("daily", limit=2)
        latest_ts = rows[0]["generated_at"]
        assert db.last_run("daily") == latest_ts


# ---------------------------------------------------------------------------
# get_recent_track_ids
# ---------------------------------------------------------------------------

class TestGetRecentTrackIds:
    def test_returns_empty_set_when_no_history(self, db):
        result = db.get_recent_playlist_track_ids("daily", 5)
        assert result == set()

    def test_returns_union_of_n_most_recent(self, db):
        db.save_playlist("daily", ["a", "b"])
        db.save_playlist("daily", ["c", "d"])
        db.save_playlist("daily", ["e", "f"])
        # n=2: last two playlists → {c, d, e, f}
        result = db.get_recent_playlist_track_ids("daily", n=2)
        assert result == {"c", "d", "e", "f"}

    def test_does_not_include_older_playlists(self, db):
        db.save_playlist("daily", ["old1", "old2"])
        db.save_playlist("daily", ["new1"])
        result = db.get_recent_playlist_track_ids("daily", n=1)
        assert result == {"new1"}
        assert "old1" not in result

    def test_type_isolation(self, db):
        db.save_playlist("daily",  ["d1"])
        db.save_playlist("weekly", ["w1"])
        daily_ids  = db.get_recent_playlist_track_ids("daily",  n=5)
        weekly_ids = db.get_recent_playlist_track_ids("weekly", n=5)
        assert daily_ids  == {"d1"}
        assert weekly_ids == {"w1"}


# ---------------------------------------------------------------------------
# genre weights
# ---------------------------------------------------------------------------

class TestGenreWeights:
    def test_get_genre_weight_unknown_returns_half(self, db):
        assert db.get_genre_weight("unknown-genre") == 0.5

    def test_get_genre_weight_empty_string_returns_half(self, db):
        assert db.get_genre_weight("") == 0.5

    def test_refresh_and_get_genre_weights(self, db):
        tracks = [
            make_track(id="a::t1", genre="hip-hop", play_count=100),
            make_track(id="a::t2", genre="hip-hop", play_count=100),  # same genre — totals 200
            make_track(id="a::t3", genre="jazz",    play_count=50),
            make_track(id="a::t4", genre="rock",    play_count=200),
        ]
        db.refresh_genre_weights(tracks)

        # rock has max plays (200) → weight 1.0
        assert db.get_genre_weight("rock") == pytest.approx(1.0)
        # hip-hop: 200/200 → 1.0
        assert db.get_genre_weight("hip-hop") == pytest.approx(1.0)
        # jazz: 50/200 → 0.25
        assert db.get_genre_weight("jazz") == pytest.approx(0.25)

    def test_refresh_genre_weights_proportional(self, db):
        tracks = [
            make_track(id="a::t1", genre="A", play_count=10),
            make_track(id="a::t2", genre="B", play_count=5),
        ]
        db.refresh_genre_weights(tracks)
        weight_a = db.get_genre_weight("A")
        weight_b = db.get_genre_weight("B")
        # A has max plays → weight=1.0; B is half → weight=0.5
        assert weight_a == pytest.approx(1.0)
        assert weight_b == pytest.approx(0.5)

    def test_refresh_genre_weights_empty_list_noop(self, db):
        db.refresh_genre_weights([])
        assert db.get_genre_weight("hip-hop") == 0.5

    def test_refresh_genre_weights_skips_no_genre(self, db):
        tracks = [
            make_track(id="a::t1", genre=None, play_count=100),
            make_track(id="a::t2", genre="",   play_count=100),
        ]
        db.refresh_genre_weights(tracks)
        # Nothing inserted — unknown genre still returns 0.5
        assert db.get_genre_weight("") == 0.5

    def test_refresh_genre_weights_upserts(self, db):
        db.refresh_genre_weights([make_track(id="a::t1", genre="pop", play_count=10)])
        db.refresh_genre_weights([make_track(id="a::t1", genre="pop", play_count=20)])
        # After second refresh, pop is still the max → weight=1.0
        assert db.get_genre_weight("pop") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_all_tracks
# ---------------------------------------------------------------------------

class TestGetAllTracks:
    def test_get_all_tracks_empty(self, db):
        assert db.get_all_tracks() == []

    def test_get_all_tracks_returns_all(self, db):
        for i in range(5):
            db.upsert_track(make_track(id=f"a{i}::t{i}"))
        tracks = db.get_all_tracks()
        assert len(tracks) == 5

    def test_get_all_tracks_deserializes_audio_features(self, db):
        db.upsert_track(make_track(audio_features={"bpm": 140.0}))
        tracks = db.get_all_tracks()
        assert isinstance(tracks[0]["audio_features"], dict)
        assert tracks[0]["audio_features"]["bpm"] == 140.0
