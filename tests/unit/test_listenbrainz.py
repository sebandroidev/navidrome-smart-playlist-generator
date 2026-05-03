"""
Unit tests for ingestion/listenbrainz.py

Covers: _norm_key, ListenBrainzClient.get_top_recordings,
        ListenBrainzClient.get_similar_recordings, enrich_tracks.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from ingestion.listenbrainz import _norm_key, ListenBrainzClient, enrich_tracks
from config import AppConfig, ListenBrainzConfig

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_track


# ---------------------------------------------------------------------------
# _norm_key
# ---------------------------------------------------------------------------

class TestNormKey:
    def test_basic_normalisation(self):
        assert _norm_key("Daft Punk", "Get Lucky") == "daftpunk::getlucky"

    def test_lowercased(self):
        assert _norm_key("ARTIST", "TITLE") == "artist::title"

    def test_strips_non_alnum(self):
        assert _norm_key("A-ha", "Take On Me") == "aha::takeonme"

    def test_strips_accents(self):
        result = _norm_key("Beyoncé", "Crazy in Love")
        assert result == "beyonce::crazyinlove"

    def test_strips_spaces(self):
        result = _norm_key("The Beatles", "Let It Be")
        assert result == "thebeatles::letitbe"

    def test_empty_artist_returns_empty(self):
        result = _norm_key("", "Some Song")
        assert result == ""

    def test_empty_title_returns_empty(self):
        result = _norm_key("Artist", "")
        assert result == ""

    def test_both_empty_returns_empty(self):
        assert _norm_key("", "") == ""

    def test_numbers_kept(self):
        result = _norm_key("2Pac", "California Love")
        assert result == "2pac::californialove"

    def test_separator_is_double_colon(self):
        result = _norm_key("Artist", "Title")
        assert "::" in result

    def test_only_separates_artist_from_title(self):
        result = _norm_key("My Artist", "My Title")
        parts = result.split("::")
        assert len(parts) == 2

    def test_unicode_normalisation(self):
        # NFKD: ﬀ → ff
        result = _norm_key("ﬀ artist", "title")
        assert "ff" in result

    def test_matches_pipeline_norm(self):
        # _norm_key must produce the same key as pipeline._track_id for same input
        from pipeline import _track_id
        artist, title = "Daft Punk", "One More Time"
        assert _norm_key(artist, title) == _track_id(artist, title)


# ---------------------------------------------------------------------------
# ListenBrainzClient.get_top_recordings
# ---------------------------------------------------------------------------

class TestGetTopRecordings:
    def _make_client(self):
        return ListenBrainzClient("testuser")

    def _mock_response(self, recordings):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "payload": {"recordings": recordings}
        }
        return resp

    def test_returns_list_on_success(self):
        client = self._make_client()
        recordings = [
            {"track_name": "Track A", "artist_name": "Artist A",
             "listen_count": 10, "recording_mbid": "mbid1"},
        ]
        with patch.object(client._session, "get",
                          return_value=self._mock_response(recordings)):
            result = client.get_top_recordings("month")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_result_contains_required_keys(self):
        client = self._make_client()
        recordings = [{"track_name": "T", "artist_name": "A",
                       "listen_count": 5, "recording_mbid": "m1"}]
        with patch.object(client._session, "get",
                          return_value=self._mock_response(recordings)):
            result = client.get_top_recordings("all_time")
        rec = result[0]
        assert "track_name" in rec
        assert "artist_name" in rec
        assert "listen_count" in rec
        assert "recording_mbid" in rec

    def test_listen_count_is_int(self):
        client = self._make_client()
        recordings = [{"track_name": "T", "artist_name": "A",
                       "listen_count": 42, "recording_mbid": "m"}]
        with patch.object(client._session, "get",
                          return_value=self._mock_response(recordings)):
            result = client.get_top_recordings("month")
        assert result[0]["listen_count"] == 42

    def test_empty_recordings_returns_empty(self):
        client = self._make_client()
        with patch.object(client._session, "get",
                          return_value=self._mock_response([])):
            result = client.get_top_recordings("month")
        assert result == []

    def test_network_error_returns_empty(self):
        client = self._make_client()
        with patch.object(client._session, "get",
                          side_effect=Exception("Connection timeout")):
            result = client.get_top_recordings("month")
        assert result == []

    def test_requests_correct_time_range(self):
        client = self._make_client()
        with patch.object(client._session, "get",
                          return_value=self._mock_response([])) as mock_get:
            client.get_top_recordings("all_time", 500)
        params = mock_get.call_args[1]["params"]
        assert params["range"] == "all_time"
        assert params["count"] == 500

    def test_missing_payload_returns_empty(self):
        client = self._make_client()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {}
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_top_recordings("month")
        assert result == []


# ---------------------------------------------------------------------------
# ListenBrainzClient.get_similar_recordings
# ---------------------------------------------------------------------------

class TestGetSimilarRecordings:
    def _make_client(self):
        return ListenBrainzClient("testuser")

    def _mock_response(self, recording_list):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "payload": {"recording_list": recording_list}
        }
        return resp

    def test_returns_list_on_success(self):
        client = self._make_client()
        recs = [{"artist_name": "A", "track_name": "T"}]
        with patch.object(client._session, "get",
                          return_value=self._mock_response(recs)):
            result = client.get_similar_recordings()
        assert isinstance(result, list)

    def test_result_items_have_track_and_artist(self):
        client = self._make_client()
        recs = [{"artist_name": "Artist A", "track_name": "Track B"}]
        with patch.object(client._session, "get",
                          return_value=self._mock_response(recs)):
            result = client.get_similar_recordings()
        assert result[0]["artist_name"] == "Artist A"
        assert result[0]["track_name"] == "Track B"

    def test_network_error_returns_empty(self):
        client = self._make_client()
        with patch.object(client._session, "get",
                          side_effect=Exception("Timeout")):
            result = client.get_similar_recordings()
        assert result == []

    def test_non_list_payload_returns_empty(self):
        client = self._make_client()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"payload": {"recording_list": "not-a-list"}}
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_similar_recordings()
        assert result == []

    def test_empty_recording_list_returns_empty(self):
        client = self._make_client()
        with patch.object(client._session, "get",
                          return_value=self._mock_response([])):
            result = client.get_similar_recordings()
        assert result == []


# ---------------------------------------------------------------------------
# enrich_tracks
# ---------------------------------------------------------------------------

class TestEnrichTracks:
    def _cfg(self, username=""):
        cfg = AppConfig()
        cfg.listenbrainz = ListenBrainzConfig(
            enabled=bool(username),
            username=username,
        )
        return cfg

    def test_empty_username_returns_unchanged(self):
        tracks = [make_track(0)]
        cfg = self._cfg(username="")
        result = enrich_tracks(tracks, cfg)
        assert result is tracks

    def test_empty_username_does_not_call_lb(self):
        tracks = [make_track(0)]
        cfg = self._cfg(username="")
        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            enrich_tracks(tracks, cfg)
        mock_cls.assert_not_called()

    def test_with_username_injects_lb_listen_count(self):
        tracks = [make_track(0, artist="Daft Punk", title="Get Lucky")]
        cfg = self._cfg(username="testuser")

        monthly = [{"artist_name": "Daft Punk", "track_name": "Get Lucky",
                    "listen_count": 42, "recording_mbid": "m1"}]

        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            instance.get_top_recordings.side_effect = [monthly, []]
            instance.get_similar_recordings.return_value = []
            result = enrich_tracks(tracks, cfg)

        assert result[0]["lb_listen_count"] == 42

    def test_with_username_injects_lb_cf_rec_false_when_no_cf(self):
        tracks = [make_track(0, artist="Daft Punk", title="Get Lucky")]
        cfg = self._cfg(username="testuser")

        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            instance.get_top_recordings.side_effect = [[], []]
            instance.get_similar_recordings.return_value = []
            result = enrich_tracks(tracks, cfg)

        assert result[0]["lb_cf_rec"] is False

    def test_cf_rec_true_when_track_in_cf_list(self):
        tracks = [make_track(0, artist="Daft Punk", title="Get Lucky")]
        cfg = self._cfg(username="testuser")

        cf_recs = [{"artist_name": "Daft Punk", "track_name": "Get Lucky"}]

        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            instance.get_top_recordings.side_effect = [[], []]
            instance.get_similar_recordings.return_value = cf_recs
            result = enrich_tracks(tracks, cfg)

        assert result[0]["lb_cf_rec"] is True

    def test_takes_max_of_monthly_and_alltime(self):
        tracks = [make_track(0, artist="Daft Punk", title="Get Lucky")]
        cfg = self._cfg(username="testuser")

        monthly = [{"artist_name": "Daft Punk", "track_name": "Get Lucky",
                    "listen_count": 50, "recording_mbid": "m"}]
        alltime = [{"artist_name": "Daft Punk", "track_name": "Get Lucky",
                    "listen_count": 200, "recording_mbid": "m"}]

        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            # alltime + monthly order: side_effect[0]=monthly, side_effect[1]=alltime
            instance.get_top_recordings.side_effect = [monthly, alltime]
            instance.get_similar_recordings.return_value = []
            result = enrich_tracks(tracks, cfg)

        assert result[0]["lb_listen_count"] == max(50, 200)

    def test_unmatched_track_gets_zero_lb_count(self):
        tracks = [make_track(0, artist="Unknown Artist", title="Unknown Track")]
        cfg = self._cfg(username="testuser")

        monthly = [{"artist_name": "Daft Punk", "track_name": "Get Lucky",
                    "listen_count": 100, "recording_mbid": "m"}]

        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            instance.get_top_recordings.side_effect = [monthly, []]
            instance.get_similar_recordings.return_value = []
            result = enrich_tracks(tracks, cfg)

        assert result[0]["lb_listen_count"] == 0

    def test_all_tracks_get_lb_keys_injected(self):
        tracks = [make_track(i) for i in range(5)]
        cfg = self._cfg(username="testuser")

        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            instance.get_top_recordings.side_effect = [[], []]
            instance.get_similar_recordings.return_value = []
            result = enrich_tracks(tracks, cfg)

        for t in result:
            assert "lb_listen_count" in t
            assert "lb_cf_rec" in t

    def test_returns_same_list_object_with_username(self):
        tracks = [make_track(0)]
        cfg = self._cfg(username="testuser")

        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            instance.get_top_recordings.side_effect = [[], []]
            instance.get_similar_recordings.return_value = []
            result = enrich_tracks(tracks, cfg)

        assert result is tracks

    def test_fetches_both_month_and_all_time(self):
        tracks = [make_track(0)]
        cfg = self._cfg(username="testuser")

        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            instance.get_top_recordings.side_effect = [[], []]
            instance.get_similar_recordings.return_value = []
            enrich_tracks(tracks, cfg)

        calls = instance.get_top_recordings.call_args_list
        ranges_called = [c[0][0] for c in calls]
        assert "month" in ranges_called
        assert "all_time" in ranges_called

    def test_lb_api_failure_still_injects_zero_values(self):
        tracks = [make_track(0)]
        cfg = self._cfg(username="testuser")

        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            # Simulate API failures — methods return []
            instance.get_top_recordings.return_value = []
            instance.get_similar_recordings.return_value = []
            result = enrich_tracks(tracks, cfg)

        assert result[0]["lb_listen_count"] == 0
        assert result[0]["lb_cf_rec"] is False

    def test_case_insensitive_matching(self):
        # norm_key lowercases; "daft punk" and "DAFT PUNK" match
        tracks = [make_track(0, artist="daft punk", title="get lucky")]
        cfg = self._cfg(username="testuser")

        monthly = [{"artist_name": "DAFT PUNK", "track_name": "GET LUCKY",
                    "listen_count": 77, "recording_mbid": "m"}]

        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            instance.get_top_recordings.side_effect = [monthly, []]
            instance.get_similar_recordings.return_value = []
            result = enrich_tracks(tracks, cfg)

        assert result[0]["lb_listen_count"] == 77

    def test_empty_tracks_returns_empty(self):
        cfg = self._cfg(username="testuser")
        with patch("ingestion.listenbrainz.ListenBrainzClient") as mock_cls:
            instance = mock_cls.return_value
            instance.get_top_recordings.side_effect = [[], []]
            instance.get_similar_recordings.return_value = []
            result = enrich_tracks([], cfg)
        assert result == []
