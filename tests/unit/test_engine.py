"""
Unit tests for scoring/engine.py  —  score_tracks()

Strategy:
  - Mock StateDB via unittest.mock.MagicMock
  - Build AppConfig with default ScoringWeights so expected composite values
    can be computed independently
  - Verify:
      * tracks modified in-place and returned
      * composite_score and signal_breakdown keys present
      * db.refresh_genre_weights called once with full track list
      * db.get_genre_weight called per track with correct genre arg
      * db.update_score called per track with correct (id, score) args
      * composite_score == weighted sum of individual signals
      * empty list short-circuits (no db calls)
"""
import math
import pytest
from unittest.mock import MagicMock, call, patch
from datetime import datetime, timezone

from config import AppConfig, ScoringConfig, ScoringWeights
from scoring.engine import score_tracks
from scoring.signals import (
    play_count_signal, recency_signal, rating_signal,
    genre_affinity_signal, discovery_bonus_signal, listenbrainz_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_track(i=0, **kw):
    defaults = {
        "id":              f"artist{i}::track{i}",
        "nav_id":          f"nav{i}",
        "title":           f"Track {i}",
        "artist":          f"Artist {i}",
        "genre":           "hip-hop",
        "year":            2020,
        "bitrate":         320,
        "bpm":             120,
        "play_count":      i,
        "last_played":     None,
        "user_rating":     0,
        "starred":         False,
        "composite_score": 0.5,
        "lb_listen_count": 0,
        "audio_features":  None,
    }
    return {**defaults, **kw}


def make_mock_db(genre_weight: float = 0.5):
    db = MagicMock()
    db.get_genre_weight.return_value = genre_weight
    return db


def make_cfg(weights: ScoringWeights | None = None, halflife: float = 7.0) -> AppConfig:
    cfg = AppConfig()
    cfg.scoring = ScoringConfig(
        weights=weights or ScoringWeights(),
        recency_halflife_days=halflife,
    )
    return cfg


def expected_score(track: dict, max_plays: int, max_lb: int,
                   genre_w: float, halflife: float,
                   w: ScoringWeights) -> float:
    pc = track.get("play_count") or 0
    s_play  = play_count_signal(pc, max_plays)
    s_rec   = recency_signal(track.get("last_played"), halflife)
    s_rate  = rating_signal(track.get("user_rating") or 0)
    s_genre = genre_affinity_signal(genre_w)
    s_disc  = discovery_bonus_signal(pc)
    s_lb    = listenbrainz_signal(track.get("lb_listen_count") or 0, max_lb)
    return (
        s_play  * w.play_count +
        s_rec   * w.recency +
        s_rate  * w.rating +
        s_genre * w.genre_affinity +
        s_disc  * w.discovery_bonus +
        s_lb    * w.lb_boost
    )


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

class TestScoreTracksBasic:
    def test_empty_tracks_returns_empty_no_db_calls(self):
        db = make_mock_db()
        cfg = make_cfg()
        result = score_tracks([], cfg, db)
        assert result == []
        db.refresh_genre_weights.assert_not_called()
        db.get_genre_weight.assert_not_called()
        db.update_score.assert_not_called()

    def test_returns_same_list_object(self):
        tracks = [make_track(1)]
        db = make_mock_db()
        cfg = make_cfg()
        result = score_tracks(tracks, cfg, db)
        assert result is tracks

    def test_composite_score_key_added(self):
        tracks = [make_track(0)]
        db = make_mock_db()
        score_tracks(tracks, make_cfg(), db)
        assert "composite_score" in tracks[0]

    def test_signal_breakdown_key_added(self):
        tracks = [make_track(0)]
        db = make_mock_db()
        score_tracks(tracks, make_cfg(), db)
        breakdown = tracks[0]["signal_breakdown"]
        assert set(breakdown.keys()) == {
            "play_count", "recency", "rating", "genre", "discovery", "lb_boost"
        }

    def test_composite_score_rounded_to_6_decimals(self):
        tracks = [make_track(3, play_count=3)]
        db = make_mock_db()
        score_tracks(tracks, make_cfg(), db)
        score = tracks[0]["composite_score"]
        # Must equal its own round to 6 places
        assert score == round(score, 6)

    def test_signal_breakdown_values_rounded_to_4_decimals(self):
        tracks = [make_track(1, play_count=1)]
        db = make_mock_db()
        score_tracks(tracks, make_cfg(), db)
        for key, val in tracks[0]["signal_breakdown"].items():
            assert val == round(val, 4), f"{key} not rounded to 4 dp"


# ---------------------------------------------------------------------------
# DB interactions
# ---------------------------------------------------------------------------

class TestScoreTracksDbCalls:
    def test_refresh_genre_weights_called_once_with_all_tracks(self):
        tracks = [make_track(i) for i in range(3)]
        db = make_mock_db()
        score_tracks(tracks, make_cfg(), db)
        db.refresh_genre_weights.assert_called_once_with(tracks)

    def test_update_score_called_for_every_track(self):
        tracks = [make_track(i) for i in range(4)]
        db = make_mock_db()
        score_tracks(tracks, make_cfg(), db)
        assert db.update_score.call_count == 4

    def test_update_score_receives_correct_id_and_score(self):
        t = make_track(7, play_count=7, genre="rock")
        db = make_mock_db(genre_weight=0.8)
        cfg = make_cfg()
        score_tracks([t], cfg, db)
        score_arg = db.update_score.call_args[0][1]
        assert score_arg == t["composite_score"]
        assert db.update_score.call_args[0][0] == "artist7::track7"

    def test_get_genre_weight_called_with_correct_genre(self):
        t = make_track(0, genre="jazz")
        db = make_mock_db()
        score_tracks([t], make_cfg(), db)
        db.get_genre_weight.assert_called_once_with("jazz")

    def test_missing_genre_skips_get_genre_weight(self):
        t = make_track(0, genre="")
        db = make_mock_db()
        score_tracks([t], make_cfg(), db)
        db.get_genre_weight.assert_not_called()

    def test_none_genre_skips_get_genre_weight(self):
        t = make_track(0, genre=None)
        db = make_mock_db()
        score_tracks([t], make_cfg(), db)
        db.get_genre_weight.assert_not_called()


# ---------------------------------------------------------------------------
# Composite score arithmetic
# ---------------------------------------------------------------------------

class TestScoreTracksComposite:
    def test_composite_matches_manual_calculation(self):
        w = ScoringWeights()
        t = make_track(0, play_count=5, user_rating=3, genre="pop",
                        lb_listen_count=2)
        genre_w = 0.6
        db = make_mock_db(genre_weight=genre_w)
        cfg = make_cfg()
        tracks = [make_track(1, play_count=10), t]  # two tracks so max_plays=10
        score_tracks(tracks, cfg, db)

        max_plays    = 10
        max_lb_plays = max(t2.get("lb_listen_count") or 0 for t2 in tracks) or 1
        exp = expected_score(t, max_plays, max_lb_plays, genre_w, 7.0, w)
        assert t["composite_score"] == pytest.approx(round(exp, 6), abs=1e-6)

    def test_never_played_track_gets_full_discovery_bonus(self):
        t = make_track(0, play_count=0)
        db = make_mock_db()
        score_tracks([t], make_cfg(), db)
        assert t["signal_breakdown"]["discovery"] == pytest.approx(1.0)

    def test_highly_played_track_gets_zero_discovery_bonus(self):
        tracks = [make_track(0, play_count=100), make_track(1, play_count=5)]
        db = make_mock_db()
        score_tracks(tracks, make_cfg(), db)
        assert tracks[0]["signal_breakdown"]["discovery"] == 0.0

    def test_rated_track_has_nonzero_rating_signal(self):
        t = make_track(0, user_rating=4)
        db = make_mock_db()
        score_tracks([t], make_cfg(), db)
        assert t["signal_breakdown"]["rating"] == pytest.approx(0.8)

    def test_lb_signal_included_in_breakdown(self):
        t = make_track(0, lb_listen_count=10)
        db = make_mock_db()
        score_tracks([t], make_cfg(), db)
        assert "lb_boost" in t["signal_breakdown"]
        assert t["signal_breakdown"]["lb_boost"] >= 0.0

    def test_lb_boost_zero_when_no_lb_data(self):
        t = make_track(0, lb_listen_count=0)
        db = make_mock_db()
        score_tracks([t], make_cfg(), db)
        assert t["signal_breakdown"]["lb_boost"] == 0.0

    def test_max_plays_derived_from_track_list(self):
        # When only one track, max_plays = max(play_count, 1)
        t = make_track(0, play_count=0)
        db = make_mock_db()
        # Should not raise ZeroDivisionError
        score_tracks([t], make_cfg(), db)
        assert 0.0 <= t["composite_score"] <= 1.0 + 1e-9

    def test_multiple_tracks_scored_independently(self):
        tracks = [make_track(i, play_count=i * 10) for i in range(5)]
        db = make_mock_db()
        score_tracks(tracks, make_cfg(), db)
        scores = [t["composite_score"] for t in tracks]
        # All scores should be valid floats
        assert all(isinstance(s, float) for s in scores)
        # Scores should differ across tracks (varying play counts)
        assert len(set(scores)) > 1

    def test_all_weights_sum_to_approximately_one(self):
        # Sanity check on the default weight config
        w = ScoringWeights()
        total = (w.play_count + w.recency + w.rating +
                 w.genre_affinity + w.discovery_bonus + w.lb_boost)
        assert total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestScoreTracksEdgeCases:
    def test_none_play_count_treated_as_zero(self):
        t = make_track(0, play_count=None)
        db = make_mock_db()
        score_tracks([t], make_cfg(), db)
        assert t["signal_breakdown"]["play_count"] == 0.0

    def test_none_user_rating_treated_as_zero(self):
        t = make_track(0, user_rating=None)
        db = make_mock_db()
        score_tracks([t], make_cfg(), db)
        assert t["signal_breakdown"]["rating"] == 0.0

    def test_none_lb_listen_count_treated_as_zero(self):
        t = make_track(0, lb_listen_count=None)
        db = make_mock_db()
        score_tracks([t], make_cfg(), db)
        assert t["signal_breakdown"]["lb_boost"] == 0.0

    def test_single_track_with_max_everything(self):
        t = make_track(0,
                       play_count=999,
                       user_rating=5,
                       genre="jazz",
                       lb_listen_count=999,
                       last_played=datetime.now(timezone.utc).isoformat())
        db = make_mock_db(genre_weight=1.0)
        cfg = make_cfg()
        score_tracks([t], cfg, db)
        # discovery_bonus_signal(999) == 0.0 (only 1.0 for play_count=0, 0.3 for 1-2)
        # max score = 1.0 - (discovery_bonus weight 0.08 × 1.0) = 0.92
        assert t["composite_score"] == pytest.approx(0.92, abs=0.01)
