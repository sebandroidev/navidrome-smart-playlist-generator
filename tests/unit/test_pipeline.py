"""
Unit tests for pipeline.py

Covers: _norm, _track_id, _merge, ingest_and_score (mocked deps).
"""
import pytest
from unittest.mock import MagicMock, patch, call

from pipeline import _norm, _track_id, _merge

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_track


# ---------------------------------------------------------------------------
# _norm
# ---------------------------------------------------------------------------

class TestNorm:
    def test_lowercases(self):
        assert _norm("HELLO") == "hello"

    def test_strips_spaces(self):
        assert _norm("hello world") == "helloworld"

    def test_strips_punctuation(self):
        assert _norm("Daft Punk") == "daftpunk"

    def test_strips_accents(self):
        # é → e after NFKD + ASCII encoding
        assert _norm("Beyoncé") == "beyonce"

    def test_strips_special_chars(self):
        assert _norm("A-ha") == "aha"

    def test_empty_string(self):
        assert _norm("") == ""

    def test_none_becomes_empty(self):
        assert _norm(None) == ""

    def test_numbers_kept(self):
        assert _norm("2Pac") == "2pac"

    def test_only_symbols_becomes_empty(self):
        assert _norm("!@#$%") == ""

    def test_unicode_ligature(self):
        # ﬀ → ff after NFKD
        result = _norm("ﬀ")
        assert result == "ff"

    def test_daft_punk_get_lucky(self):
        assert _norm("Daft Punk") == "daftpunk"

    def test_japanese_stripped(self):
        # Non-ASCII chars are stripped to empty after encode+ignore
        result = _norm("テスト")
        assert result == ""


# ---------------------------------------------------------------------------
# _track_id
# ---------------------------------------------------------------------------

class TestTrackId:
    def test_combines_artist_and_title(self):
        assert _track_id("Daft Punk", "Get Lucky") == "daftpunk::getlucky"

    def test_separator_is_double_colon(self):
        result = _track_id("Artist", "Title")
        assert "::" in result

    def test_both_normalised(self):
        result = _track_id("Beyoncé", "Crazy in Love")
        assert result == "beyonce::crazyinlove"

    def test_empty_artist(self):
        result = _track_id("", "Song")
        assert result == "::song"

    def test_empty_title(self):
        result = _track_id("Artist", "")
        assert result == "artist::"

    def test_both_empty(self):
        assert _track_id("", "") == "::"


# ---------------------------------------------------------------------------
# _merge — nav songs as source of truth
# ---------------------------------------------------------------------------

def _nav_song(i, artist="Artist", title="Track", **kw):
    defaults = {
        "id": f"nav{i}",
        "artist": artist,
        "title": title,
        "album": f"Album {i}",
        "genre": None,
        "year": None,
        "playCount": 0,
        "lastPlayed": None,
        "played": None,
        "starred": False,
        "userRating": 0,
        "bitRate": None,
        "suffix": "mp3",
        "albumArtist": artist,
    }
    return {**defaults, **kw}


def _beets_track(artist="Artist", title="Track", **kw):
    defaults = {
        "artist": artist,
        "title": title,
        "genre": "rock",
        "year": 2020,
        "bitrate": 320,
        "format": "MP3",
        "path": "/music/song.mp3",
        "beets_id": 1,
        "album": "Some Album",
    }
    return {**defaults, **kw}


class TestMerge:
    def test_nav_songs_without_artist_skipped(self):
        songs = [_nav_song(0, artist="", title="Track")]
        result = _merge(songs, [])
        assert result == []

    def test_nav_songs_without_title_skipped(self):
        songs = [_nav_song(0, artist="Artist", title="")]
        result = _merge(songs, [])
        assert result == []

    def test_nav_only_track_included(self):
        songs = [_nav_song(0, artist="Bob", title="Song")]
        result = _merge(songs, [])
        assert len(result) == 1
        assert result[0]["nav_id"] == "nav0"

    def test_beets_enriches_genre(self):
        songs = [_nav_song(0, artist="Bob", title="Song")]
        bt = [_beets_track(artist="Bob", title="Song", genre="jazz")]
        result = _merge(songs, bt)
        assert result[0]["genre"] == "jazz"

    def test_nav_genre_takes_priority_over_beets(self):
        songs = [_nav_song(0, artist="Bob", title="Song", genre="hip-hop")]
        bt = [_beets_track(artist="Bob", title="Song", genre="jazz")]
        result = _merge(songs, bt)
        # nav genre is non-None → takes priority
        assert result[0]["genre"] == "hip-hop"

    def test_beets_enriches_bitrate(self):
        songs = [_nav_song(0, artist="Bob", title="Song")]
        bt = [_beets_track(artist="Bob", title="Song", bitrate=320)]
        result = _merge(songs, bt)
        assert result[0]["bitrate"] == 320

    def test_nav_bitrate_fallback(self):
        songs = [_nav_song(0, artist="Bob", title="Song", bitRate=192)]
        result = _merge(songs, [])
        assert result[0]["bitrate"] == 192

    def test_beets_enriches_path(self):
        songs = [_nav_song(0, artist="Bob", title="Song")]
        bt = [_beets_track(artist="Bob", title="Song", path="/my/song.mp3")]
        result = _merge(songs, bt)
        assert result[0]["path"] == "/my/song.mp3"

    def test_nav_id_preserved(self):
        songs = [_nav_song(42, artist="Alice", title="Song")]
        result = _merge(songs, [])
        assert result[0]["nav_id"] == "nav42"

    def test_play_count_from_nav(self):
        songs = [_nav_song(0, artist="Alice", title="Song", playCount=7)]
        result = _merge(songs, [])
        assert result[0]["play_count"] == 7

    def test_play_count_defaults_to_zero(self):
        songs = [_nav_song(0, artist="Alice", title="Song", playCount=None)]
        result = _merge(songs, [])
        assert result[0]["play_count"] == 0

    def test_starred_from_nav(self):
        songs = [_nav_song(0, artist="Alice", title="Song", starred=True)]
        result = _merge(songs, [])
        assert result[0]["starred"] is True

    def test_user_rating_from_nav(self):
        songs = [_nav_song(0, artist="Alice", title="Song", userRating=4)]
        result = _merge(songs, [])
        assert result[0]["user_rating"] == 4

    def test_composite_score_initialised_to_zero(self):
        songs = [_nav_song(0, artist="Alice", title="Song")]
        result = _merge(songs, [])
        assert result[0]["composite_score"] == 0.0

    def test_audio_features_initialised_to_none(self):
        songs = [_nav_song(0, artist="Alice", title="Song")]
        result = _merge(songs, [])
        assert result[0]["audio_features"] is None

    def test_duplicate_nav_songs_deduplicated(self):
        songs = [
            _nav_song(0, artist="Alice", title="Song"),
            _nav_song(1, artist="Alice", title="Song"),  # same artist+title
        ]
        result = _merge(songs, [])
        assert len(result) == 1

    def test_multiple_nav_songs_all_included(self):
        songs = [
            _nav_song(0, artist="Alice", title="Song A"),
            _nav_song(1, artist="Bob",   title="Song B"),
            _nav_song(2, artist="Carol", title="Song C"),
        ]
        result = _merge(songs, [])
        assert len(result) == 3

    def test_unmatched_beets_track_ignored(self):
        songs = [_nav_song(0, artist="Alice", title="Song A")]
        bt = [_beets_track(artist="Zara", title="Unknown Track")]
        result = _merge(songs, bt)
        assert len(result) == 1
        assert result[0]["artist"] == "Alice"

    def test_normalisation_matches_across_artist_title(self):
        # "Daft Punk" → "daftpunk"; beets has same spelling
        songs = [_nav_song(0, artist="Daft Punk", title="Get Lucky")]
        bt = [_beets_track(artist="Daft Punk", title="Get Lucky", genre="house", bitrate=320)]
        result = _merge(songs, bt)
        assert result[0]["genre"] == "house"

    def test_last_played_from_played_field(self):
        played_iso = "2024-01-15T12:00:00+00:00"
        songs = [_nav_song(0, artist="A", title="B", played=played_iso)]
        result = _merge(songs, [])
        assert result[0]["last_played"] == played_iso

    def test_last_played_from_lastPlayed_epoch(self):
        songs = [_nav_song(0, artist="A", title="B", lastPlayed=1700000000)]
        result = _merge(songs, [])
        assert result[0]["last_played"] is not None
        # Should be an ISO string
        assert "T" in result[0]["last_played"]

    def test_last_played_none_when_no_play_data(self):
        songs = [_nav_song(0, artist="A", title="B", played=None, lastPlayed=None)]
        result = _merge(songs, [])
        assert result[0]["last_played"] is None

    def test_result_has_all_required_keys(self):
        songs = [_nav_song(0, artist="A", title="B")]
        result = _merge(songs, [])
        required = {"id", "nav_id", "title", "artist", "genre", "bitrate",
                    "play_count", "last_played", "starred", "user_rating",
                    "composite_score", "audio_features"}
        assert required.issubset(result[0].keys())

    def test_empty_nav_returns_empty(self):
        assert _merge([], []) == []

    def test_empty_beets_still_merges_nav(self):
        songs = [_nav_song(0, artist="X", title="Y")]
        result = _merge(songs, [])
        assert len(result) == 1

    def test_beets_format_enrichment(self):
        songs = [_nav_song(0, artist="A", title="B")]
        bt = [_beets_track(artist="A", title="B", format="FLAC")]
        result = _merge(songs, bt)
        assert result[0]["format"] == "FLAC"

    def test_nav_suffix_fallback_when_no_beets_format(self):
        songs = [_nav_song(0, artist="A", title="B", suffix="ogg")]
        result = _merge(songs, [])
        assert result[0]["format"] == "ogg"


# ---------------------------------------------------------------------------
# ingest_and_score — mocked pipeline
# ---------------------------------------------------------------------------

class TestIngestAndScore:
    def _run(self, nav_songs, beets_tracks, cfg=None, db=None):
        from config import AppConfig
        from pipeline import ingest_and_score
        if cfg is None:
            cfg = AppConfig()
            cfg.listenbrainz.enabled = False
            cfg.audio_analysis.enabled = False
        if db is None:
            db = MagicMock()

        with patch("pipeline._fetch_nav_songs", return_value=nav_songs), \
             patch("pipeline._fetch_beets_tracks", return_value=beets_tracks), \
             patch("pipeline.score_tracks") as mock_score:
            # score_tracks modifies tracks in place
            mock_score.side_effect = lambda t, c, d: t
            result = ingest_and_score(cfg, db)

        return result, mock_score

    def test_returns_list_of_dicts(self):
        songs = [_nav_song(0, artist="A", title="B")]
        result, _ = self._run(songs, [])
        assert isinstance(result, list)
        assert all(isinstance(t, dict) for t in result)

    def test_score_tracks_called_with_merged_list(self):
        songs = [_nav_song(0, artist="A", title="B")]
        result, mock_score = self._run(songs, [])
        mock_score.assert_called_once()

    def test_upsert_track_called_for_each_track(self):
        from pipeline import ingest_and_score
        from config import AppConfig
        cfg = AppConfig()
        cfg.listenbrainz.enabled = False
        cfg.audio_analysis.enabled = False
        db = MagicMock()

        songs = [_nav_song(i, artist=f"A{i}", title=f"T{i}") for i in range(3)]
        with patch("pipeline._fetch_nav_songs", return_value=songs), \
             patch("pipeline._fetch_beets_tracks", return_value=[]), \
             patch("pipeline.score_tracks", side_effect=lambda t, c, d: t):
            ingest_and_score(cfg, db)

        assert db.upsert_track.call_count == 3

    def test_nav_cache_invalidated_before_fetch(self):
        from pipeline import ingest_and_score
        from config import AppConfig
        import ingestion.cache as cache_mod
        cfg = AppConfig()
        cfg.listenbrainz.enabled = False
        cfg.audio_analysis.enabled = False
        db = MagicMock()

        cache_mod.set("nav_songs", [{"old": True}])

        with patch("pipeline._fetch_nav_songs", return_value=[]) as mock_fetch, \
             patch("pipeline._fetch_beets_tracks", return_value=[]), \
             patch("pipeline.score_tracks", side_effect=lambda t, c, d: t):
            ingest_and_score(cfg, db)

        # _fetch_nav_songs should have been called (re-fetching)
        mock_fetch.assert_called_once()
