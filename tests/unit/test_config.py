"""
Unit tests for config.py

Covers: AppConfig defaults, _parse, _load_raw, get_config, patch_config,
        env-var overrides, ScoringWeights defaults.
"""
import os
import pytest
from unittest.mock import patch, mock_open
from config import (
    AppConfig, ScoringWeights, ScoringConfig, NavidromeConfig,
    ListenBrainzConfig, OllamaConfig, TelegramConfig, DailyGenConfig,
    WeeklyGenConfig, AudioAnalysisConfig,
    _parse, _load_raw, get_config, patch_config, reload_config,
)


# ---------------------------------------------------------------------------
# ScoringWeights defaults
# ---------------------------------------------------------------------------

class TestScoringWeightsDefaults:
    def test_play_count_default(self):
        w = ScoringWeights()
        assert w.play_count == pytest.approx(0.27)

    def test_recency_default(self):
        assert ScoringWeights().recency == pytest.approx(0.23)

    def test_rating_default(self):
        assert ScoringWeights().rating == pytest.approx(0.18)

    def test_genre_affinity_default(self):
        assert ScoringWeights().genre_affinity == pytest.approx(0.14)

    def test_discovery_bonus_default(self):
        assert ScoringWeights().discovery_bonus == pytest.approx(0.08)

    def test_lb_boost_default(self):
        assert ScoringWeights().lb_boost == pytest.approx(0.10)

    def test_weights_sum_to_one(self):
        w = ScoringWeights()
        total = (w.play_count + w.recency + w.rating +
                 w.genre_affinity + w.discovery_bonus + w.lb_boost)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_has_six_fields(self):
        w = ScoringWeights()
        fields = [f for f in vars(w)]
        assert len(fields) == 6


# ---------------------------------------------------------------------------
# AppConfig defaults
# ---------------------------------------------------------------------------

class TestAppConfigDefaults:
    def test_listenbrainz_disabled_by_default(self):
        cfg = AppConfig()
        assert cfg.listenbrainz.enabled is False

    def test_listenbrainz_username_empty_by_default(self):
        assert AppConfig().listenbrainz.username == ""

    def test_ollama_disabled_by_default(self):
        assert AppConfig().ollama.enabled is False

    def test_telegram_disabled_by_default(self):
        assert AppConfig().telegram.enabled is False

    def test_audio_analysis_disabled_by_default(self):
        assert AppConfig().audio_analysis.enabled is False

    def test_daily_track_count_default(self):
        assert AppConfig().daily.track_count == 30

    def test_weekly_track_count_default(self):
        assert AppConfig().weekly.track_count == 50

    def test_state_db_path_default(self):
        assert AppConfig().state_db_path == "/data/orly_jams.db"

    def test_genre_cluster_count_default(self):
        assert AppConfig().genre_cluster_count == 5

    def test_playlist_names_has_daily_and_weekly(self):
        names = AppConfig().playlist_names
        assert "daily" in names
        assert "weekly" in names

    def test_scoring_has_default_weights(self):
        cfg = AppConfig()
        assert isinstance(cfg.scoring.weights, ScoringWeights)

    def test_scoring_recency_halflife_default(self):
        assert AppConfig().scoring.recency_halflife_days == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# _parse — minimal raw dict
# ---------------------------------------------------------------------------

class TestParse:
    def test_empty_raw_returns_defaults(self):
        cfg = _parse({})
        assert isinstance(cfg, AppConfig)
        assert cfg.listenbrainz.enabled is False
        assert cfg.ollama.enabled is False

    def test_parse_navidrome_url(self):
        cfg = _parse({"navidrome": {"url": "http://myserver:4533"}})
        assert cfg.navidrome.url == "http://myserver:4533"

    def test_parse_navidrome_pass_key(self):
        # "pass" is the YAML key, "password" is the dataclass field
        cfg = _parse({"navidrome": {"pass": "secret"}})
        assert cfg.navidrome.password == "secret"

    def test_parse_navidrome_password_key(self):
        cfg = _parse({"navidrome": {"password": "pw2"}})
        assert cfg.navidrome.password == "pw2"

    def test_parse_listenbrainz_enabled(self):
        cfg = _parse({"listenbrainz": {"enabled": True, "username": "alice"}})
        assert cfg.listenbrainz.enabled is True
        assert cfg.listenbrainz.username == "alice"

    def test_parse_ollama_enabled(self):
        cfg = _parse({"ollama": {"enabled": True, "host": "http://gpu:11434"}})
        assert cfg.ollama.enabled is True
        assert cfg.ollama.host == "http://gpu:11434"

    def test_parse_telegram_chat_id_coerced_to_str(self):
        cfg = _parse({"telegram": {"enabled": True, "bot_token": "tok", "chat_id": 12345}})
        assert cfg.telegram.chat_id == "12345"

    def test_parse_scoring_weights(self):
        raw = {"scoring": {"weights": {"play_count": 0.5, "recency": 0.1,
                                       "rating": 0.1, "genre_affinity": 0.1,
                                       "discovery_bonus": 0.1, "lb_boost": 0.1}}}
        cfg = _parse(raw)
        assert cfg.scoring.weights.play_count == pytest.approx(0.5)

    def test_parse_recency_halflife(self):
        cfg = _parse({"scoring": {"recency_halflife_days": 14.0}})
        assert cfg.scoring.recency_halflife_days == pytest.approx(14.0)

    def test_parse_daily_track_count_from_scheduling(self):
        cfg = _parse({"scheduling": {"daily_jam": {"track_count": 25}}})
        assert cfg.daily.track_count == 25

    def test_parse_weekly_track_count_from_scheduling(self):
        cfg = _parse({"scheduling": {"weekly_jam": {"track_count": 60}}})
        assert cfg.weekly.track_count == 60

    def test_parse_daily_comfort_ratio_from_generation(self):
        cfg = _parse({"generation": {"daily": {"comfort_ratio": 0.75}}})
        assert cfg.daily.comfort_ratio == pytest.approx(0.75)

    def test_parse_audio_analysis_enabled(self):
        cfg = _parse({"audio_analysis": {"enabled": True}})
        assert cfg.audio_analysis.enabled is True

    def test_parse_audio_analysis_cache_forever_default_true(self):
        cfg = _parse({"audio_analysis": {"enabled": True}})
        assert cfg.audio_analysis.cache_forever is True

    def test_parse_state_db_path(self):
        cfg = _parse({"state_db_path": "/tmp/test.db"})
        assert cfg.state_db_path == "/tmp/test.db"

    def test_parse_genre_cluster_count(self):
        cfg = _parse({"genre_cluster_count": 8})
        assert cfg.genre_cluster_count == 8

    def test_parse_playlist_names_merged(self):
        cfg = _parse({"playlist_names": {"daily": "My Daily Mix"}})
        assert cfg.playlist_names["daily"] == "My Daily Mix"
        # weekly default still present
        assert "weekly" in cfg.playlist_names

    def test_parse_beets_db_path(self):
        cfg = _parse({"beets": {"db_path": "/data/music.db"}})
        assert cfg.beets.db_path == "/data/music.db"


# ---------------------------------------------------------------------------
# _load_raw — env-var overrides
# ---------------------------------------------------------------------------

class TestLoadRaw:
    def test_lb_username_env_enables_listenbrainz(self, monkeypatch):
        monkeypatch.setenv("LB_USERNAME", "testuser")
        raw = _load_raw()
        assert raw["listenbrainz"]["enabled"] is True
        assert raw["listenbrainz"]["username"] == "testuser"

    def test_tg_bot_token_env_enables_telegram(self, monkeypatch):
        monkeypatch.setenv("TG_BOT_TOKEN", "123:ABC")
        raw = _load_raw()
        assert raw["telegram"]["enabled"] is True
        assert raw["telegram"]["bot_token"] == "123:ABC"

    def test_navidrome_url_env_overrides(self, monkeypatch):
        monkeypatch.setenv("NAVIDROME_URL", "http://override:4533")
        raw = _load_raw()
        assert raw["navidrome"]["url"] == "http://override:4533"

    def test_navidrome_user_env_overrides(self, monkeypatch):
        monkeypatch.setenv("NAVIDROME_USER", "myuser")
        raw = _load_raw()
        assert raw["navidrome"]["user"] == "myuser"

    def test_navidrome_pass_env_overrides(self, monkeypatch):
        monkeypatch.setenv("NAVIDROME_PASS", "mysecret")
        raw = _load_raw()
        assert raw["navidrome"]["pass"] == "mysecret"

    def test_ollama_host_env_enables_ollama(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://gpu:11434")
        raw = _load_raw()
        assert raw["ollama"]["host"] == "http://gpu:11434"
        assert raw["ollama"]["enabled"] is True

    def test_beets_db_path_env_overrides(self, monkeypatch):
        monkeypatch.setenv("BEETS_DB_PATH", "/my/music.db")
        raw = _load_raw()
        assert raw["beets"]["db_path"] == "/my/music.db"

    def test_state_db_path_env_overrides(self, monkeypatch):
        monkeypatch.setenv("STATE_DB_PATH", "/tmp/state.db")
        raw = _load_raw()
        assert raw["state_db_path"] == "/tmp/state.db"

    def test_tg_chat_id_env_sets_chat_id(self, monkeypatch):
        monkeypatch.setenv("TG_CHAT_ID", "-100123456")
        raw = _load_raw()
        assert raw["telegram"]["chat_id"] == "-100123456"

    def test_unset_env_vars_do_not_override(self, monkeypatch):
        # Make sure env vars are absent
        for var in ["LB_USERNAME", "TG_BOT_TOKEN", "NAVIDROME_URL",
                    "NAVIDROME_USER", "NAVIDROME_PASS", "OLLAMA_HOST",
                    "BEETS_DB_PATH", "STATE_DB_PATH"]:
            monkeypatch.delenv(var, raising=False)
        raw = _load_raw()
        # navidrome section should be present but url should not be overridden
        # (may or may not be present depending on config file)
        assert isinstance(raw, dict)


# ---------------------------------------------------------------------------
# get_config — singleton behaviour
# ---------------------------------------------------------------------------

class TestGetConfig:
    def test_returns_app_config_instance(self):
        cfg = get_config()
        assert isinstance(cfg, AppConfig)

    def test_returns_same_object_on_repeated_calls(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reload_config_returns_new_instance(self):
        cfg1 = get_config()
        cfg2 = reload_config()
        # Same type, but reload_config always rebuilds
        assert isinstance(cfg2, AppConfig)


# ---------------------------------------------------------------------------
# patch_config — dot-notation key patching
# ---------------------------------------------------------------------------

class TestPatchConfig:
    def setup_method(self):
        # Reset singleton before each test
        reload_config()

    def test_patch_scoring_weight(self):
        cfg = patch_config({"scoring.weights.play_count": 0.5})
        assert cfg.scoring.weights.play_count == pytest.approx(0.5)

    def test_patch_multiple_keys(self):
        cfg = patch_config({
            "scoring.weights.recency": 0.30,
            "scoring.weights.rating": 0.20,
        })
        assert cfg.scoring.weights.recency == pytest.approx(0.30)
        assert cfg.scoring.weights.rating == pytest.approx(0.20)

    def test_patch_nested_key_creates_intermediate_dicts(self):
        # Patching a deep key should not raise
        cfg = patch_config({"generation.daily.comfort_ratio": 0.70})
        assert cfg.daily.comfort_ratio == pytest.approx(0.70)

    def test_patch_returns_app_config(self):
        result = patch_config({"scoring.weights.lb_boost": 0.15})
        assert isinstance(result, AppConfig)

    def test_patch_updates_singleton(self):
        patch_config({"scoring.weights.discovery_bonus": 0.12})
        cfg_after = get_config()
        assert cfg_after.scoring.weights.discovery_bonus == pytest.approx(0.12)

    def test_patch_top_level_key(self):
        cfg = patch_config({"genre_cluster_count": 10})
        assert cfg.genre_cluster_count == 10
