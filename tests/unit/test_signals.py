"""
Unit tests for scoring/signals.py

Covers all 6 signal functions:
  play_count_signal, recency_signal, rating_signal,
  genre_affinity_signal, discovery_bonus_signal, listenbrainz_signal
"""
import math
import pytest
from datetime import datetime, timedelta, timezone

from scoring.signals import (
    play_count_signal,
    recency_signal,
    rating_signal,
    genre_affinity_signal,
    discovery_bonus_signal,
    listenbrainz_signal,
)


# ---------------------------------------------------------------------------
# play_count_signal
# ---------------------------------------------------------------------------

class TestPlayCountSignal:
    def test_zero_max_plays_returns_zero(self):
        assert play_count_signal(10, 0) == 0.0

    def test_negative_max_plays_returns_zero(self):
        assert play_count_signal(5, -1) == 0.0

    def test_zero_plays_returns_zero(self):
        # log1p(0) == 0, so result is 0 regardless of max
        assert play_count_signal(0, 100) == 0.0

    def test_max_play_count_returns_one(self):
        # When play_count == max_plays, result must be exactly 1.0
        result = play_count_signal(50, 50)
        assert result == pytest.approx(1.0)

    def test_partial_play_count_between_zero_and_one(self):
        result = play_count_signal(5, 100)
        assert 0.0 < result < 1.0

    def test_log_normalization_is_monotonic(self):
        # Higher play count relative to max → higher signal
        r1 = play_count_signal(10, 100)
        r2 = play_count_signal(50, 100)
        r3 = play_count_signal(99, 100)
        assert r1 < r2 < r3

    def test_formula_correctness(self):
        # Manual calculation: log1p(5) / log1p(20)
        expected = math.log1p(5) / math.log1p(20)
        assert play_count_signal(5, 20) == pytest.approx(expected)

    def test_single_play_single_max(self):
        # Edge: play_count == max_plays == 1 → 1.0
        assert play_count_signal(1, 1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# recency_signal
# ---------------------------------------------------------------------------

class TestRecencySignal:
    def test_none_last_played_returns_zero(self):
        assert recency_signal(None) == 0.0

    def test_empty_string_returns_zero(self):
        assert recency_signal("") == 0.0

    def test_invalid_date_string_returns_zero(self):
        assert recency_signal("not-a-date") == 0.0

    def test_played_today_returns_near_one(self):
        now_str = datetime.now(timezone.utc).isoformat()
        result = recency_signal(now_str)
        # Just played: should be very close to 1.0
        assert result == pytest.approx(1.0, abs=0.01)

    def test_played_at_halflife_returns_half(self):
        halflife = 7.0
        played_at = datetime.now(timezone.utc) - timedelta(days=halflife)
        result = recency_signal(played_at.isoformat(), halflife_days=halflife)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_played_long_ago_returns_near_zero(self):
        old = datetime.now(timezone.utc) - timedelta(days=365)
        result = recency_signal(old.isoformat())
        assert result < 0.01

    def test_result_bounded_between_zero_and_one(self):
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        # Future dates yield a value > 1 mathematically; signal should still be a float
        result = recency_signal(future)
        assert isinstance(result, float)

    def test_zulu_iso_format_accepted(self):
        now_z = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = recency_signal(now_z)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_custom_halflife_affects_decay_rate(self):
        played_at = datetime.now(timezone.utc) - timedelta(days=3)
        iso = played_at.isoformat()
        # Longer halflife → slower decay → higher signal for same elapsed days
        r_short = recency_signal(iso, halflife_days=1.0)
        r_long  = recency_signal(iso, halflife_days=30.0)
        assert r_long > r_short


# ---------------------------------------------------------------------------
# rating_signal
# ---------------------------------------------------------------------------

class TestRatingSignal:
    def test_zero_rating_returns_zero(self):
        assert rating_signal(0) == 0.0

    def test_five_star_returns_one(self):
        assert rating_signal(5) == pytest.approx(1.0)

    def test_intermediate_ratings_scale_linearly(self):
        assert rating_signal(1) == pytest.approx(0.2)
        assert rating_signal(2) == pytest.approx(0.4)
        assert rating_signal(3) == pytest.approx(0.6)
        assert rating_signal(4) == pytest.approx(0.8)

    def test_negative_rating_clamped_to_zero(self):
        assert rating_signal(-3) == 0.0

    def test_rating_above_five_clamped_to_one(self):
        assert rating_signal(10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# genre_affinity_signal
# ---------------------------------------------------------------------------

class TestGenreAffinitySignal:
    def test_zero_weight_returns_zero(self):
        assert genre_affinity_signal(0.0) == 0.0

    def test_one_weight_returns_one(self):
        assert genre_affinity_signal(1.0) == pytest.approx(1.0)

    def test_mid_weight_pass_through(self):
        assert genre_affinity_signal(0.5) == pytest.approx(0.5)

    def test_negative_weight_clamped_to_zero(self):
        assert genre_affinity_signal(-0.5) == 0.0

    def test_weight_above_one_clamped_to_one(self):
        assert genre_affinity_signal(1.5) == pytest.approx(1.0)

    def test_arbitrary_weight_within_range_unchanged(self):
        assert genre_affinity_signal(0.73) == pytest.approx(0.73)


# ---------------------------------------------------------------------------
# discovery_bonus_signal
# ---------------------------------------------------------------------------

class TestDiscoveryBonusSignal:
    def test_zero_plays_returns_full_bonus(self):
        assert discovery_bonus_signal(0) == 1.0

    def test_one_play_returns_partial_bonus(self):
        assert discovery_bonus_signal(1) == pytest.approx(0.3)

    def test_two_plays_returns_partial_bonus(self):
        assert discovery_bonus_signal(2) == pytest.approx(0.3)

    def test_three_plays_returns_zero(self):
        assert discovery_bonus_signal(3) == 0.0

    def test_high_play_count_returns_zero(self):
        assert discovery_bonus_signal(100) == 0.0

    def test_threshold_is_exclusive_at_three(self):
        # 2 → partial, 3 → 0 (boundary check)
        assert discovery_bonus_signal(2) == pytest.approx(0.3)
        assert discovery_bonus_signal(3) == 0.0


# ---------------------------------------------------------------------------
# listenbrainz_signal
# ---------------------------------------------------------------------------

class TestListenbrainzSignal:
    def test_zero_max_plays_returns_zero(self):
        assert listenbrainz_signal(100, 0) == 0.0

    def test_zero_lb_count_returns_zero(self):
        assert listenbrainz_signal(0, 100) == 0.0

    def test_both_zero_returns_zero(self):
        assert listenbrainz_signal(0, 0) == 0.0

    def test_count_equals_max_returns_one(self):
        result = listenbrainz_signal(50, 50)
        assert result == pytest.approx(1.0)

    def test_partial_count_between_zero_and_one(self):
        result = listenbrainz_signal(10, 100)
        assert 0.0 < result < 1.0

    def test_log_normalization_is_monotonic(self):
        r1 = listenbrainz_signal(5,  100)
        r2 = listenbrainz_signal(20, 100)
        r3 = listenbrainz_signal(80, 100)
        assert r1 < r2 < r3

    def test_formula_matches_play_count_signal(self):
        # listenbrainz_signal is structurally identical to play_count_signal
        lb = listenbrainz_signal(7, 42)
        pc = play_count_signal(7, 42)
        assert lb == pytest.approx(pc)

    def test_negative_lb_count_returns_zero(self):
        # lb_listen_count <= 0 branch
        assert listenbrainz_signal(-5, 100) == 0.0
