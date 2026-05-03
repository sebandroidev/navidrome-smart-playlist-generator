"""
Shared pytest fixtures and helpers for unit and integration tests.
"""
import os
import pytest

from config import AppConfig, ScoringConfig, ScoringWeights, DailyGenConfig, WeeklyGenConfig
from config import ListenBrainzConfig, OllamaConfig, TelegramConfig, NavidromeConfig
from state.db import StateDB


# ---------------------------------------------------------------------------
# Track factory — used directly by tests (not a fixture)
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


# ---------------------------------------------------------------------------
# sample_tracks helper — returns n tracks with varied attributes
# ---------------------------------------------------------------------------

_GENRES = ["hip-hop", "jazz", "rock", "pop", "electronic", "classical", "ambient", "folk"]


def sample_tracks(n=30):
    tracks = []
    for i in range(n):
        tracks.append(make_track(
            i,
            genre=_GENRES[i % len(_GENRES)],
            play_count=i * 2,
            composite_score=round(0.1 + (i % 10) * 0.08, 2),
            bpm=60 + (i * 7) % 100,
        ))
    return tracks


# ---------------------------------------------------------------------------
# test_cfg fixture — AppConfig with listenbrainz + ollama disabled
# ---------------------------------------------------------------------------

@pytest.fixture
def test_cfg():
    cfg = AppConfig()
    cfg.listenbrainz = ListenBrainzConfig(enabled=False, username="")
    cfg.ollama = OllamaConfig(enabled=False)
    cfg.telegram = TelegramConfig(enabled=False, bot_token="", chat_id="")
    cfg.daily = DailyGenConfig(
        track_count=30,
        comfort_ratio=0.60,
        exclude_played_within_hours=48,
    )
    cfg.weekly = WeeklyGenConfig(
        track_count=50,
        comfort_ratio=0.40,
        exclude_last_n_weekly_playlists=2,
    )
    cfg.scoring = ScoringConfig(weights=ScoringWeights(), recency_halflife_days=7.0)
    return cfg


# ---------------------------------------------------------------------------
# test_db fixture — real StateDB backed by a temp file (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("db")
    db_path = str(tmp / "test_state.db")
    db = StateDB(db_path)
    return db
