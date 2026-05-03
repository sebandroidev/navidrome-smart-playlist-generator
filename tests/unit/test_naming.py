"""
Unit tests for generation/naming.py

Covers: rule_based_name, ollama_name (mocked HTTP), generate_name.
"""
import pytest
from unittest.mock import patch, MagicMock

from config import AppConfig, OllamaConfig
from generation.naming import rule_based_name, ollama_name, generate_name

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_ollama_off():
    cfg = AppConfig()
    cfg.ollama = OllamaConfig(enabled=False)
    return cfg


def _cfg_ollama_on(host="http://localhost:11434", model="gemma2:2b"):
    cfg = AppConfig()
    cfg.ollama = OllamaConfig(enabled=True, host=host, model=model)
    return cfg


# ---------------------------------------------------------------------------
# rule_based_name
# ---------------------------------------------------------------------------

class TestRuleBasedName:
    def test_daily_contains_daily_jam(self):
        name = rule_based_name("daily", [make_track(0, genre="hip-hop")])
        assert "Daily Jam" in name

    def test_weekly_contains_weekly_jam(self):
        name = rule_based_name("weekly", [make_track(0, genre="jazz")])
        assert "Weekly Jam" in name

    def test_daily_contains_genre(self):
        name = rule_based_name("daily", [make_track(0, genre="jazz")])
        assert "jazz" in name.lower()

    def test_weekly_contains_week_number(self):
        name = rule_based_name("weekly", [make_track(0)])
        assert "week" in name.lower()

    def test_empty_tracks_uses_mixed_genre(self):
        name = rule_based_name("daily", [])
        assert "mixed" in name.lower()

    def test_no_genre_falls_back_to_mixed(self):
        name = rule_based_name("daily", [make_track(0, genre=None)])
        assert "mixed" in name.lower()

    def test_returns_string(self):
        assert isinstance(rule_based_name("daily", [make_track(0)]), str)

    def test_up_to_two_genres_shown(self):
        tracks = [
            make_track(0, genre="rock"),
            make_track(1, genre="jazz"),
            make_track(2, genre="pop"),
        ]
        name = rule_based_name("daily", tracks)
        # Only up to 2 genres; "pop" (3rd) should not appear
        assert "pop" not in name

    def test_duplicate_genres_deduplicated(self):
        tracks = [make_track(i, genre="rock") for i in range(5)]
        name = rule_based_name("daily", tracks)
        # "rock" should appear only once
        assert name.lower().count("rock") == 1

    def test_daily_name_has_day_and_slot(self):
        name = rule_based_name("daily", [make_track(0, genre="pop")])
        days = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]
        slots = ["morning", "afternoon", "evening", "night", "late night"]
        lower = name.lower()
        assert any(d in lower for d in days)
        assert any(s in lower for s in slots)

    def test_weekly_name_emoji(self):
        name = rule_based_name("weekly", [make_track(0)])
        assert "🗓" in name

    def test_daily_name_emoji(self):
        name = rule_based_name("daily", [make_track(0)])
        assert "🎵" in name


# ---------------------------------------------------------------------------
# ollama_name — mocked HTTP
# ---------------------------------------------------------------------------

class TestOllamaName:
    def _mock_response(self, text):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": text}
        return mock_resp

    def test_uses_ollama_response_as_name(self):
        cfg = _cfg_ollama_on()
        tracks = [make_track(0, genre="hip-hop")]
        with patch("generation.naming.requests.post",
                   return_value=self._mock_response("sunday vibes")) as mock_post:
            name = ollama_name("daily", tracks, cfg)
        assert "sunday vibes" in name.lower()
        mock_post.assert_called_once()

    def test_daily_prefix_added_to_ollama_response(self):
        cfg = _cfg_ollama_on()
        with patch("generation.naming.requests.post",
                   return_value=self._mock_response("cool grooves")):
            name = ollama_name("daily", [make_track(0)], cfg)
        assert name.startswith("🎵")

    def test_weekly_prefix_added_to_ollama_response(self):
        cfg = _cfg_ollama_on()
        with patch("generation.naming.requests.post",
                   return_value=self._mock_response("weekend anthems")):
            name = ollama_name("weekly", [make_track(0)], cfg)
        assert name.startswith("🗓")

    def test_empty_ollama_response_falls_back_to_rule_based(self):
        cfg = _cfg_ollama_on()
        with patch("generation.naming.requests.post",
                   return_value=self._mock_response("")):
            name = ollama_name("daily", [make_track(0, genre="pop")], cfg)
        # Should fall back to rule_based_name result
        assert "Daily Jam" in name

    def test_ollama_http_error_falls_back_to_rule_based(self):
        cfg = _cfg_ollama_on()
        with patch("generation.naming.requests.post",
                   side_effect=Exception("Connection refused")):
            name = ollama_name("daily", [make_track(0, genre="rock")], cfg)
        assert "Daily Jam" in name

    def test_strips_quotes_from_ollama_response(self):
        cfg = _cfg_ollama_on()
        with patch("generation.naming.requests.post",
                   return_value=self._mock_response('"quoted name"')):
            name = ollama_name("daily", [make_track(0)], cfg)
        assert '"' not in name

    def test_posts_to_correct_host(self):
        cfg = _cfg_ollama_on(host="http://myhost:11434")
        with patch("generation.naming.requests.post",
                   return_value=self._mock_response("vibes")) as mock_post:
            ollama_name("daily", [make_track(0)], cfg)
        call_url = mock_post.call_args[0][0]
        assert "myhost:11434" in call_url

    def test_posts_with_correct_model(self):
        cfg = _cfg_ollama_on(model="llama3:8b")
        with patch("generation.naming.requests.post",
                   return_value=self._mock_response("rhythm")) as mock_post:
            ollama_name("daily", [make_track(0)], cfg)
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "llama3:8b"

    def test_raise_for_status_called(self):
        cfg = _cfg_ollama_on()
        mock_resp = self._mock_response("some name")
        with patch("generation.naming.requests.post", return_value=mock_resp):
            ollama_name("daily", [make_track(0)], cfg)
        mock_resp.raise_for_status.assert_called_once()


# ---------------------------------------------------------------------------
# generate_name — dispatcher
# ---------------------------------------------------------------------------

class TestGenerateName:
    def test_ollama_disabled_uses_rule_based(self):
        cfg = _cfg_ollama_off()
        with patch("generation.naming.ollama_name") as mock_ol:
            name = generate_name("daily", [make_track(0, genre="jazz")], cfg)
        mock_ol.assert_not_called()
        assert "Daily Jam" in name

    def test_ollama_enabled_calls_ollama_name(self):
        cfg = _cfg_ollama_on()
        with patch("generation.naming.ollama_name",
                   return_value="🎵 custom name") as mock_ol:
            name = generate_name("daily", [make_track(0)], cfg)
        mock_ol.assert_called_once()
        assert name == "🎵 custom name"

    def test_returns_string(self):
        cfg = _cfg_ollama_off()
        name = generate_name("weekly", [make_track(0)], cfg)
        assert isinstance(name, str)

    def test_weekly_rule_based_when_ollama_off(self):
        cfg = _cfg_ollama_off()
        name = generate_name("weekly", [make_track(0, genre="soul")], cfg)
        assert "Weekly Jam" in name
