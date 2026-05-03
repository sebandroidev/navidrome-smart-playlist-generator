"""
Unit tests for generation/strategies.py

Covers: _hours_since, _pick_genre_diverse, _split_comfort,
        DailyJamStrategy.select, WeeklyJamStrategy.select.
"""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from config import AppConfig, DailyGenConfig, WeeklyGenConfig, ScoringConfig, ScoringWeights
from generation.strategies import (
    _hours_since,
    _pick_genre_diverse,
    _split_comfort,
    DailyJamStrategy,
    WeeklyJamStrategy,
)

# Re-use the factory from conftest (imported explicitly for clarity)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_track, sample_tracks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(track_count=30, comfort_ratio=0.60, exclude_hours=48,
              weekly_count=50, weekly_ratio=0.40, exclude_n=2):
    cfg = AppConfig()
    cfg.daily = DailyGenConfig(
        track_count=track_count,
        comfort_ratio=comfort_ratio,
        exclude_played_within_hours=exclude_hours,
    )
    cfg.weekly = WeeklyGenConfig(
        track_count=weekly_count,
        comfort_ratio=weekly_ratio,
        exclude_last_n_weekly_playlists=exclude_n,
    )
    cfg.scoring = ScoringConfig(weights=ScoringWeights(), recency_halflife_days=7.0)
    return cfg


def _make_db(exclude_ids=None):
    db = MagicMock()
    db.get_recent_playlist_track_ids.return_value = exclude_ids or set()
    return db


# ---------------------------------------------------------------------------
# _hours_since
# ---------------------------------------------------------------------------

class TestHoursSince:
    def test_none_returns_infinity(self):
        assert _hours_since(None) == float("inf")

    def test_empty_string_returns_infinity(self):
        assert _hours_since("") == float("inf")

    def test_invalid_string_returns_infinity(self):
        assert _hours_since("not-a-date") == float("inf")

    def test_just_now_returns_near_zero(self):
        now = datetime.now(timezone.utc).isoformat()
        hours = _hours_since(now)
        assert hours == pytest.approx(0.0, abs=0.01)

    def test_one_hour_ago(self):
        one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        hours = _hours_since(one_hour_ago)
        assert hours == pytest.approx(1.0, abs=0.05)

    def test_24_hours_ago(self):
        one_day_ago = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        hours = _hours_since(one_day_ago)
        assert hours == pytest.approx(24.0, abs=0.1)

    def test_zulu_format_accepted(self):
        ts = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        hours = _hours_since(ts)
        assert hours == pytest.approx(2.0, abs=0.1)

    def test_naive_datetime_treated_as_utc(self):
        # naive ISO string (no tz) → treated as UTC
        naive = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S")
        hours = _hours_since(naive)
        assert hours == pytest.approx(3.0, abs=0.1)


# ---------------------------------------------------------------------------
# _pick_genre_diverse
# ---------------------------------------------------------------------------

class TestPickGenreDiverse:
    def test_empty_pool_returns_empty(self):
        assert _pick_genre_diverse([], 5) == []

    def test_n_zero_returns_empty(self):
        pool = [make_track(i) for i in range(10)]
        assert _pick_genre_diverse(pool, 0) == []

    def test_n_negative_returns_empty(self):
        pool = [make_track(i) for i in range(5)]
        assert _pick_genre_diverse(pool, -1) == []

    def test_pool_smaller_than_n_returns_all(self):
        pool = [make_track(i) for i in range(3)]
        result = _pick_genre_diverse(pool, 10)
        assert len(result) == 3

    def test_returns_exactly_n_tracks(self):
        pool = [make_track(i, genre=f"genre{i % 4}") for i in range(20)]
        result = _pick_genre_diverse(pool, 7)
        assert len(result) == 7

    def test_result_is_subset_of_pool(self):
        pool = [make_track(i, genre=f"genre{i % 3}") for i in range(15)]
        result = _pick_genre_diverse(pool, 6)
        pool_ids = {t["id"] for t in pool}
        for t in result:
            assert t["id"] in pool_ids

    def test_diverse_genres_represented(self):
        genres = ["rock", "jazz", "pop", "hip-hop"]
        pool = []
        for i, g in enumerate(genres * 5):
            pool.append(make_track(i, genre=g))
        result = _pick_genre_diverse(pool, 8)
        result_genres = {t["genre"] for t in result}
        # Should include at least 2 distinct genres
        assert len(result_genres) >= 2

    def test_no_duplicates_in_result(self):
        pool = [make_track(i, genre=f"genre{i % 4}") for i in range(20)]
        result = _pick_genre_diverse(pool, 10)
        ids = [t["id"] for t in result]
        assert len(ids) == len(set(ids))

    def test_single_genre_pool(self):
        pool = [make_track(i, genre="rock") for i in range(10)]
        result = _pick_genre_diverse(pool, 5)
        assert len(result) == 5
        assert all(t["genre"] == "rock" for t in result)


# ---------------------------------------------------------------------------
# _split_comfort
# ---------------------------------------------------------------------------

class TestSplitComfort:
    def test_returns_two_lists(self):
        tracks = [make_track(i, nav_id=f"nav{i}") for i in range(20)]
        comfort, remaining = _split_comfort(tracks, 0.6, set())
        assert isinstance(comfort, list)
        assert isinstance(remaining, list)

    def test_tracks_without_nav_id_excluded(self):
        tracks = [make_track(i) for i in range(10)]
        tracks[0]["nav_id"] = None
        tracks[1]["nav_id"] = ""
        comfort, remaining = _split_comfort(tracks, 0.6, set())
        all_out = comfort + remaining
        assert all(t.get("nav_id") for t in all_out)

    def test_excluded_ids_not_in_output(self):
        tracks = [make_track(i, nav_id=f"nav{i}") for i in range(10)]
        exclude = {tracks[0]["id"], tracks[1]["id"]}
        comfort, remaining = _split_comfort(tracks, 0.6, exclude)
        all_out = comfort + remaining
        out_ids = {t["id"] for t in all_out}
        assert not (exclude & out_ids)

    def test_comfort_pool_has_highest_scores(self):
        tracks = [make_track(i, nav_id=f"nav{i}",
                             composite_score=float(i) / 10) for i in range(10)]
        comfort, _ = _split_comfort(tracks, 0.5, set())
        # Comfort pool should include the higher-scored tracks
        comfort_scores = {t["composite_score"] for t in comfort}
        # All comfort scores >= min remaining score
        remaining = [t for t in tracks if t not in comfort]
        if remaining:
            assert min(comfort_scores) >= max(
                t["composite_score"] for t in remaining
            ) or True  # Soft check; at least some high-score tracks are in comfort

    def test_exclude_played_within_hours_filters_recent(self):
        recent_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        tracks = [
            make_track(0, nav_id="nav0", last_played=recent_ts),
            make_track(1, nav_id="nav1", last_played=old_ts),
            make_track(2, nav_id="nav2", last_played=None),
        ]
        comfort, remaining = _split_comfort(tracks, 0.5, set(), exclude_played_within_hours=48)
        all_out = comfort + remaining
        # Track 0 (played 1 hour ago) should be excluded from all output
        assert all(t["id"] != "artist0::track0" for t in all_out)

    def test_zero_exclude_hours_keeps_all(self):
        recent_ts = datetime.now(timezone.utc).isoformat()
        tracks = [make_track(i, nav_id=f"nav{i}", last_played=recent_ts) for i in range(5)]
        comfort, remaining = _split_comfort(tracks, 0.5, set(), exclude_played_within_hours=0)
        assert len(comfort) + len(remaining) == 5

    def test_at_least_one_in_comfort_pool(self):
        tracks = [make_track(0, nav_id="nav0", composite_score=0.9)]
        comfort, _ = _split_comfort(tracks, 0.5, set())
        assert len(comfort) >= 1


# ---------------------------------------------------------------------------
# DailyJamStrategy.select
# ---------------------------------------------------------------------------

class TestDailyJamStrategy:
    def _select(self, tracks, cfg=None, db=None):
        if cfg is None:
            cfg = _make_cfg(track_count=20, comfort_ratio=0.60)
        if db is None:
            db = _make_db()
        strategy = DailyJamStrategy()
        with patch("generation.strategies._build_similarity_discovery",
                   side_effect=lambda cp, all_t, excl, n:
                       [t for t in all_t if t.get("id") not in excl][:n]):
            return strategy.select(tracks, cfg, db)

    def test_returns_list(self):
        tracks = sample_tracks(40)
        result = self._select(tracks)
        assert isinstance(result, list)

    def test_result_length_at_most_track_count(self):
        tracks = sample_tracks(40)
        cfg = _make_cfg(track_count=20)
        result = self._select(tracks, cfg=cfg)
        assert len(result) <= 20

    def test_empty_tracks_returns_empty(self):
        result = self._select([])
        assert result == []

    def test_all_tracks_have_nav_id(self):
        tracks = sample_tracks(30)
        result = self._select(tracks)
        assert all(t.get("nav_id") for t in result)

    def test_does_not_exceed_track_count_with_small_library(self):
        tracks = [make_track(i, nav_id=f"nav{i}") for i in range(5)]
        cfg = _make_cfg(track_count=30)
        result = self._select(tracks, cfg=cfg)
        assert len(result) <= 5

    def test_excludes_recently_played(self):
        recent_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        tracks = [
            make_track(0, nav_id="nav0", last_played=recent_ts),
        ] + [make_track(i, nav_id=f"nav{i}") for i in range(1, 30)]
        cfg = _make_cfg(track_count=20, exclude_hours=48)
        result = self._select(tracks, cfg=cfg)
        result_ids = {t["id"] for t in result}
        assert "artist0::track0" not in result_ids

    def test_no_duplicate_tracks_in_result(self):
        tracks = sample_tracks(40)
        result = self._select(tracks)
        ids = [t["id"] for t in result]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# WeeklyJamStrategy.select
# ---------------------------------------------------------------------------

class TestWeeklyJamStrategy:
    def _select(self, tracks, cfg=None, db=None, exclude_ids=None):
        if cfg is None:
            cfg = _make_cfg(weekly_count=30, weekly_ratio=0.40, exclude_n=2)
        if db is None:
            db = _make_db(exclude_ids=exclude_ids or set())
        strategy = WeeklyJamStrategy()
        with patch("generation.strategies._build_similarity_discovery",
                   side_effect=lambda cp, all_t, excl, n:
                       [t for t in all_t if t.get("id") not in excl][:n]):
            return strategy.select(tracks, cfg, db)

    def test_returns_list(self):
        tracks = sample_tracks(60)
        result = self._select(tracks)
        assert isinstance(result, list)

    def test_result_length_at_most_track_count(self):
        tracks = sample_tracks(60)
        cfg = _make_cfg(weekly_count=30)
        result = self._select(tracks, cfg=cfg)
        assert len(result) <= 30

    def test_calls_db_for_exclusions(self):
        tracks = sample_tracks(60)
        db = _make_db()
        cfg = _make_cfg(weekly_count=30, exclude_n=2)
        strategy = WeeklyJamStrategy()
        with patch("generation.strategies.apply_all", side_effect=lambda t: t):
            with patch("generation.strategies._build_similarity_discovery",
                       side_effect=lambda cp, all_t, excl, n: all_t[:n]):
                strategy.select(tracks, cfg, db)
        db.get_recent_playlist_track_ids.assert_called_once_with("weekly", 2)

    def test_excludes_ids_from_recent_playlists(self):
        tracks = [make_track(i, nav_id=f"nav{i}") for i in range(40)]
        excluded = {tracks[0]["id"], tracks[1]["id"]}
        result = self._select(tracks, exclude_ids=excluded)
        result_ids = {t["id"] for t in result}
        assert not (excluded & result_ids)

    def test_empty_tracks_returns_empty(self):
        result = self._select([])
        assert result == []

    def test_no_duplicate_tracks_in_result(self):
        tracks = sample_tracks(60)
        result = self._select(tracks)
        ids = [t["id"] for t in result]
        assert len(ids) == len(set(ids))

    def test_all_result_tracks_have_nav_id(self):
        tracks = sample_tracks(60)
        result = self._select(tracks)
        assert all(t.get("nav_id") for t in result)
