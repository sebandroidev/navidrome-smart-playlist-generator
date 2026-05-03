"""
Unit tests for notifier.py

Covers: notify, _handle_command (/daily, /weekly, /genres, /moods, /rescan,
        /stats, /preview_daily, /preview_weekly, /help, unknown command),
        start_bot_polling.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from config import AppConfig, TelegramConfig
import notifier
from notifier import notify, _handle_command, start_bot_polling, _HELP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_tg_enabled(token="bot123:TOKEN", chat_id="-100000001"):
    cfg = AppConfig()
    cfg.telegram = TelegramConfig(enabled=True, bot_token=token, chat_id=chat_id)
    return cfg


def _cfg_tg_disabled():
    cfg = AppConfig()
    cfg.telegram = TelegramConfig(enabled=False, bot_token="", chat_id="")
    return cfg


def _mock_session_post(accepted=True, status_code=200):
    resp = MagicMock()
    resp.json.return_value = {"accepted": accepted}
    resp.status_code = status_code
    return resp


def _mock_session_get(**json_data):
    resp = MagicMock()
    resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# notify
# ---------------------------------------------------------------------------

class TestNotify:
    def test_does_not_call_requests_when_disabled(self):
        cfg = _cfg_tg_disabled()
        with patch("notifier.requests.post") as mock_post:
            notify("hello", cfg)
        mock_post.assert_not_called()

    def test_does_not_call_requests_when_no_token(self):
        cfg = AppConfig()
        cfg.telegram = TelegramConfig(enabled=True, bot_token="", chat_id="-123")
        with patch("notifier.requests.post") as mock_post:
            notify("hello", cfg)
        mock_post.assert_not_called()

    def test_does_not_call_requests_when_no_chat_id(self):
        cfg = AppConfig()
        cfg.telegram = TelegramConfig(enabled=True, bot_token="tok", chat_id="")
        with patch("notifier.requests.post") as mock_post:
            notify("hello", cfg)
        mock_post.assert_not_called()

    def test_calls_requests_post_when_enabled(self):
        cfg = _cfg_tg_enabled()
        with patch("notifier.requests.post") as mock_post:
            notify("test message", cfg)
        mock_post.assert_called_once()

    def test_posts_to_sendmessage_endpoint(self):
        cfg = _cfg_tg_enabled(token="mybot:TOKEN")
        with patch("notifier.requests.post") as mock_post:
            notify("msg", cfg)
        url = mock_post.call_args[0][0]
        assert "sendMessage" in url
        assert "mybot:TOKEN" in url

    def test_includes_message_in_post_payload(self):
        cfg = _cfg_tg_enabled()
        with patch("notifier.requests.post") as mock_post:
            notify("my notification", cfg)
        payload = mock_post.call_args[1]["json"]
        assert payload["text"] == "my notification"

    def test_includes_chat_id_in_payload(self):
        cfg = _cfg_tg_enabled(chat_id="-999")
        with patch("notifier.requests.post") as mock_post:
            notify("msg", cfg)
        payload = mock_post.call_args[1]["json"]
        assert payload["chat_id"] == "-999"

    def test_uses_markdown_parse_mode(self):
        cfg = _cfg_tg_enabled()
        with patch("notifier.requests.post") as mock_post:
            notify("*bold*", cfg)
        payload = mock_post.call_args[1]["json"]
        assert payload["parse_mode"] == "Markdown"

    def test_request_exception_does_not_raise(self):
        cfg = _cfg_tg_enabled()
        with patch("notifier.requests.post", side_effect=Exception("Network error")):
            notify("msg", cfg)  # Should not raise


# ---------------------------------------------------------------------------
# _handle_command — /daily
# ---------------------------------------------------------------------------

class TestHandleCommandDaily:
    def test_accepted_returns_started_message(self):
        sess = MagicMock()
        sess.post.return_value = _mock_session_post(accepted=True)
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/daily", "http://localhost:7070")
        assert "Daily Jam" in reply
        assert "started" in reply.lower()

    def test_not_accepted_returns_already_running(self):
        sess = MagicMock()
        sess.post.return_value = _mock_session_post(accepted=False)
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/daily", "http://localhost:7070")
        assert "running" in reply.lower() or "Already" in reply

    def test_posts_to_trigger_daily(self):
        sess = MagicMock()
        sess.post.return_value = _mock_session_post(accepted=True)
        with patch("notifier.requests.Session", return_value=sess):
            _handle_command("/daily", "http://localhost:7070")
        call_url = sess.post.call_args[0][0]
        assert "/trigger/daily" in call_url

    def test_command_with_at_suffix(self):
        # /daily@botname → cmd = "daily"
        sess = MagicMock()
        sess.post.return_value = _mock_session_post(accepted=True)
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/daily@orlybot", "http://localhost:7070")
        assert "Daily Jam" in reply

    def test_command_uppercase(self):
        sess = MagicMock()
        sess.post.return_value = _mock_session_post(accepted=True)
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/DAILY", "http://localhost:7070")
        assert "Daily Jam" in reply


# ---------------------------------------------------------------------------
# _handle_command — /weekly
# ---------------------------------------------------------------------------

class TestHandleCommandWeekly:
    def test_accepted_returns_started_message(self):
        sess = MagicMock()
        sess.post.return_value = _mock_session_post(accepted=True)
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/weekly", "http://localhost:7070")
        assert "Weekly Jam" in reply

    def test_posts_to_trigger_weekly(self):
        sess = MagicMock()
        sess.post.return_value = _mock_session_post(accepted=True)
        with patch("notifier.requests.Session", return_value=sess):
            _handle_command("/weekly", "http://localhost:7070")
        assert "/trigger/weekly" in sess.post.call_args[0][0]


# ---------------------------------------------------------------------------
# _handle_command — /genres and /moods
# ---------------------------------------------------------------------------

class TestHandleCommandGenresMoods:
    def test_genres_posts_to_trigger_clusters(self):
        sess = MagicMock()
        sess.post.return_value = _mock_session_post(accepted=True)
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/genres", "http://localhost:7070")
        assert "/trigger/clusters" in sess.post.call_args[0][0]
        assert "Genre Mixes" in reply

    def test_moods_posts_to_trigger_moods(self):
        sess = MagicMock()
        sess.post.return_value = _mock_session_post(accepted=True)
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/moods", "http://localhost:7070")
        assert "/trigger/moods" in sess.post.call_args[0][0]
        assert "Mood Mixes" in reply


# ---------------------------------------------------------------------------
# _handle_command — /rescan
# ---------------------------------------------------------------------------

class TestHandleCommandRescan:
    def test_rescan_posts_to_rescan_endpoint(self):
        sess = MagicMock()
        sess.post.return_value = MagicMock()
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/rescan", "http://localhost:7070")
        assert "/rescan" in sess.post.call_args[0][0]

    def test_rescan_returns_started_message(self):
        sess = MagicMock()
        sess.post.return_value = MagicMock()
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/rescan", "http://localhost:7070")
        assert "rescan" in reply.lower()


# ---------------------------------------------------------------------------
# _handle_command — /stats
# ---------------------------------------------------------------------------

class TestHandleCommandStats:
    def _run_stats(self, library_size=100, play_coverage_pct=75.0,
                   avg_score=0.65, top_genres=None):
        if top_genres is None:
            top_genres = [{"genre": "hip-hop"}, {"genre": "jazz"}, {"genre": "rock"}]

        health_resp = MagicMock()
        health_resp.json.return_value = {"library_size": library_size}
        stats_resp = MagicMock()
        stats_resp.json.return_value = {
            "play_coverage_pct": play_coverage_pct,
            "avg_score": avg_score,
            "top_genres": top_genres,
        }

        sess = MagicMock()
        # get is called twice: /health, /stats
        sess.get.side_effect = [health_resp, stats_resp]

        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/stats", "http://localhost:7070")
        return reply, sess

    def test_stats_calls_health_endpoint(self):
        _, sess = self._run_stats()
        urls = [c[0][0] for c in sess.get.call_args_list]
        assert any("/health" in u for u in urls)

    def test_stats_calls_stats_endpoint(self):
        _, sess = self._run_stats()
        urls = [c[0][0] for c in sess.get.call_args_list]
        assert any("/stats" in u for u in urls)

    def test_stats_includes_track_count(self):
        reply, _ = self._run_stats(library_size=250)
        assert "250" in reply

    def test_stats_includes_play_coverage(self):
        reply, _ = self._run_stats(play_coverage_pct=82.5)
        assert "82.5" in reply

    def test_stats_includes_avg_score(self):
        reply, _ = self._run_stats(avg_score=0.73)
        assert "0.73" in reply

    def test_stats_includes_top_genres(self):
        reply, _ = self._run_stats(top_genres=[{"genre": "hip-hop"}, {"genre": "jazz"}])
        assert "hip-hop" in reply

    def test_stats_top_genres_up_to_three(self):
        genres = [{"genre": f"genre{i}"} for i in range(5)]
        reply, _ = self._run_stats(top_genres=genres)
        # Only first 3 genres should appear
        assert "genre0" in reply
        assert "genre3" not in reply

    def test_stats_empty_genres_shows_dash(self):
        reply, _ = self._run_stats(top_genres=[])
        assert "—" in reply


# ---------------------------------------------------------------------------
# _handle_command — /help
# ---------------------------------------------------------------------------

class TestHandleCommandHelp:
    def test_help_returns_help_string(self):
        sess = MagicMock()
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/help", "http://localhost:7070")
        assert reply == _HELP

    def test_help_does_not_make_http_calls(self):
        sess = MagicMock()
        with patch("notifier.requests.Session", return_value=sess):
            _handle_command("/help", "http://localhost:7070")
        sess.get.assert_not_called()
        sess.post.assert_not_called()


# ---------------------------------------------------------------------------
# _handle_command — unknown command
# ---------------------------------------------------------------------------

class TestHandleCommandUnknown:
    def test_unknown_returns_error_message(self):
        sess = MagicMock()
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/unknowncmd", "http://localhost:7070")
        assert "Unknown" in reply or "unknown" in reply

    def test_unknown_includes_help_text(self):
        sess = MagicMock()
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/xyzzy", "http://localhost:7070")
        # Should include some part of _HELP
        assert "/daily" in reply or "commands" in reply.lower()


# ---------------------------------------------------------------------------
# _handle_command — /preview_daily and /preview_weekly
# ---------------------------------------------------------------------------

class TestHandleCommandPreview:
    def _preview(self, ptype, tracks=None):
        if tracks is None:
            tracks = [
                {"artist": f"Artist {i}", "title": f"Title {i}"}
                for i in range(5)
            ]
        resp = MagicMock()
        resp.json.return_value = {"tracks": tracks}
        sess = MagicMock()
        sess.get.return_value = resp
        with patch("notifier.requests.Session", return_value=sess):
            return _handle_command(f"/preview_{ptype}", "http://localhost:7070")

    def test_preview_daily_calls_preview_endpoint(self):
        resp = MagicMock()
        resp.json.return_value = {"tracks": [{"artist": "A", "title": "B"}]}
        sess = MagicMock()
        sess.get.return_value = resp
        with patch("notifier.requests.Session", return_value=sess):
            _handle_command("/preview_daily", "http://localhost:7070")
        url = sess.get.call_args[0][0]
        assert "/playlist/daily/preview" in url

    def test_preview_weekly_calls_correct_endpoint(self):
        resp = MagicMock()
        resp.json.return_value = {"tracks": [{"artist": "A", "title": "B"}]}
        sess = MagicMock()
        sess.get.return_value = resp
        with patch("notifier.requests.Session", return_value=sess):
            _handle_command("/preview_weekly", "http://localhost:7070")
        url = sess.get.call_args[0][0]
        assert "/playlist/weekly/preview" in url

    def test_preview_shows_up_to_5_tracks(self):
        reply = self._preview("daily")
        # Should contain track numbers 1 through 5
        assert "1." in reply

    def test_preview_shows_artist_and_title(self):
        reply = self._preview("daily", [{"artist": "Daft Punk", "title": "Get Lucky"}])
        assert "Daft Punk" in reply
        assert "Get Lucky" in reply

    def test_preview_empty_tracks_returns_no_preview_message(self):
        resp = MagicMock()
        resp.json.return_value = {"tracks": []}
        sess = MagicMock()
        sess.get.return_value = resp
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/preview_daily", "http://localhost:7070")
        assert "No" in reply or "no" in reply


# ---------------------------------------------------------------------------
# _handle_command — exception handling
# ---------------------------------------------------------------------------

class TestHandleCommandExceptions:
    def test_exception_during_command_returns_error_message(self):
        sess = MagicMock()
        sess.post.side_effect = Exception("Timeout")
        with patch("notifier.requests.Session", return_value=sess):
            reply = _handle_command("/daily", "http://localhost:7070")
        assert "Error" in reply or "error" in reply


# ---------------------------------------------------------------------------
# start_bot_polling
# ---------------------------------------------------------------------------

class TestStartBotPolling:
    def test_does_not_start_thread_when_disabled(self):
        cfg = _cfg_tg_disabled()
        with patch("notifier.threading.Thread") as mock_thread:
            start_bot_polling(cfg)
        mock_thread.assert_not_called()

    def test_does_not_start_thread_when_no_token(self):
        cfg = AppConfig()
        cfg.telegram = TelegramConfig(enabled=True, bot_token="", chat_id="-123")
        with patch("notifier.threading.Thread") as mock_thread:
            start_bot_polling(cfg)
        mock_thread.assert_not_called()

    def test_starts_daemon_thread_when_enabled(self):
        cfg = _cfg_tg_enabled()
        mock_thread_obj = MagicMock()
        with patch("notifier.threading.Thread", return_value=mock_thread_obj) as mock_thread:
            start_bot_polling(cfg)
        mock_thread.assert_called_once()
        mock_thread_obj.start.assert_called_once()

    def test_thread_is_daemon(self):
        cfg = _cfg_tg_enabled()
        with patch("notifier.threading.Thread") as mock_thread:
            start_bot_polling(cfg)
        kwargs = mock_thread.call_args[1]
        assert kwargs.get("daemon") is True

    def test_thread_named_tg_bot(self):
        cfg = _cfg_tg_enabled()
        with patch("notifier.threading.Thread") as mock_thread:
            start_bot_polling(cfg)
        kwargs = mock_thread.call_args[1]
        assert kwargs.get("name") == "tg-bot"
