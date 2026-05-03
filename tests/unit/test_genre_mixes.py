"""
Unit tests for generation/genre_mixes.py

Covers: run_genre_mixes — mocked cluster_tracks, NavidromeClient, db, notifier.
"""
import pytest
from unittest.mock import MagicMock, patch, call

from config import AppConfig
from generation.genre_mixes import run_genre_mixes, _MIX_SIZE

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_track, sample_tracks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cluster(cluster_id, name, genres, n_tracks=20):
    tracks = [
        make_track(cluster_id * 100 + i,
                   nav_id=f"nav_c{cluster_id}_{i}",
                   genre=genres[0] if genres else "rock",
                   composite_score=0.5 + i * 0.01)
        for i in range(n_tracks)
    ]
    return {
        "cluster_id": cluster_id,
        "name": name,
        "genres": genres,
        "tracks": tracks,
    }


def _run(clusters, cfg=None, db=None, push_return="pl_id_x"):
    if cfg is None:
        cfg = AppConfig()
    if db is None:
        db = MagicMock()

    mock_client = MagicMock()
    mock_client.push_playlist.return_value = push_return

    with patch("generation.genre_mixes.cluster_tracks", return_value=clusters) as mock_cluster, \
         patch("generation.genre_mixes.nav_mod") as mock_nav, \
         patch("generation.genre_mixes.notifier") as mock_notifier:
        mock_nav.NavidromeClient.return_value = mock_client
        results = run_genre_mixes(sample_tracks(30), cfg, db)

    return results, mock_client, db, mock_notifier


# ---------------------------------------------------------------------------
# Empty / no-cluster cases
# ---------------------------------------------------------------------------

class TestRunGenreMixesEdgeCases:
    def test_no_clusters_returns_empty(self):
        results, client, db, notifier = _run([])
        assert results == []
        client.push_playlist.assert_not_called()
        db.save_playlist.assert_not_called()
        notifier.notify.assert_not_called()

    def test_cluster_with_no_picks_skipped(self):
        # Cluster that has tracks but all are filtered out by constraints
        cluster = {
            "cluster_id": 0,
            "name": "Empty Cluster",
            "genres": ["rock"],
            "tracks": [],  # empty after constraints
        }
        results, client, _, _ = _run([cluster])
        assert results == []
        client.push_playlist.assert_not_called()

    def test_cluster_with_tracks_missing_nav_id_skipped(self):
        # filter_unmatched will drop these; picks will be empty
        tracks = [make_track(i, nav_id=None) for i in range(10)]
        cluster = {"cluster_id": 0, "name": "No Nav", "genres": ["pop"], "tracks": tracks}
        results, client, _, _ = _run([cluster])
        assert results == []


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestRunGenreMixesHappyPath:
    def test_returns_one_result_per_non_empty_cluster(self):
        clusters = [
            _make_cluster(0, "Hip Hop Mix", ["hip-hop"]),
            _make_cluster(1, "Jazz Mix",    ["jazz"]),
        ]
        results, _, _, _ = _run(clusters)
        assert len(results) == 2

    def test_result_has_required_keys(self):
        clusters = [_make_cluster(0, "Rock Mix", ["rock"])]
        results, _, _, _ = _run(clusters)
        r = results[0]
        assert "name" in r
        assert "cluster_id" in r
        assert "genres" in r
        assert "track_count" in r
        assert "nav_playlist_id" in r

    def test_track_count_at_most_mix_size(self):
        # Give each cluster more than _MIX_SIZE tracks
        clusters = [_make_cluster(0, "Big Mix", ["rock"], n_tracks=_MIX_SIZE * 3)]
        results, _, _, _ = _run(clusters)
        assert results[0]["track_count"] <= _MIX_SIZE

    def test_push_playlist_called_for_each_cluster(self):
        clusters = [
            _make_cluster(0, "Mix A", ["rock"]),
            _make_cluster(1, "Mix B", ["jazz"]),
            _make_cluster(2, "Mix C", ["pop"]),
        ]
        _, client, _, _ = _run(clusters)
        assert client.push_playlist.call_count == 3

    def test_db_save_playlist_called_as_genre_mix(self):
        clusters = [_make_cluster(0, "A Mix", ["hip-hop"])]
        _, _, db, _ = _run(clusters)
        db.save_playlist.assert_called_once()
        call_type = db.save_playlist.call_args[0][0]
        assert call_type == "genre_mix"

    def test_nav_playlist_id_in_result(self):
        clusters = [_make_cluster(0, "Nav Mix", ["rock"])]
        results, _, _, _ = _run(clusters, push_return="playlist-abc-123")
        assert results[0]["nav_playlist_id"] == "playlist-abc-123"

    def test_genres_field_in_result(self):
        clusters = [_make_cluster(0, "Multi Genre", ["hip-hop", "r&b"])]
        results, _, _, _ = _run(clusters)
        assert results[0]["genres"] == ["hip-hop", "r&b"]

    def test_cluster_id_preserved_in_result(self):
        clusters = [_make_cluster(7, "Mix Seven", ["jazz"])]
        results, _, _, _ = _run(clusters)
        assert results[0]["cluster_id"] == 7


# ---------------------------------------------------------------------------
# Duplicate name deduplication
# ---------------------------------------------------------------------------

class TestRunGenreMixesDuplicateNames:
    def test_duplicate_cluster_names_get_ii_suffix(self):
        # Two clusters with the same name
        cluster_a = _make_cluster(0, "Rock Mix", ["rock"])
        cluster_b = _make_cluster(1, "Rock Mix", ["rock", "punk"])
        results, _, _, _ = _run([cluster_a, cluster_b])
        names = [r["name"] for r in results]
        assert "Rock Mix" in names
        assert "Rock Mix II" in names

    def test_unique_cluster_names_unchanged(self):
        clusters = [
            _make_cluster(0, "Hip Hop Mix", ["hip-hop"]),
            _make_cluster(1, "Jazz Mix",    ["jazz"]),
        ]
        results, _, _, _ = _run(clusters)
        names = [r["name"] for r in results]
        assert "Hip Hop Mix" in names
        assert "Jazz Mix" in names


# ---------------------------------------------------------------------------
# Telegram notification
# ---------------------------------------------------------------------------

class TestRunGenreMixesNotifier:
    def test_notifier_called_when_results_exist(self):
        clusters = [_make_cluster(0, "Pop Mix", ["pop"])]
        _, _, _, mock_notifier = _run(clusters)
        mock_notifier.notify.assert_called_once()

    def test_notifier_message_mentions_count(self):
        clusters = [
            _make_cluster(0, "Mix A", ["rock"]),
            _make_cluster(1, "Mix B", ["jazz"]),
        ]
        _, _, _, mock_notifier = _run(clusters)
        msg = mock_notifier.notify.call_args[0][0]
        assert "2" in msg

    def test_notifier_not_called_when_no_results(self):
        _, _, _, mock_notifier = _run([])
        mock_notifier.notify.assert_not_called()


# ---------------------------------------------------------------------------
# cfg.genre_cluster_count passed to cluster_tracks
# ---------------------------------------------------------------------------

class TestRunGenreMixesClusterCount:
    def test_cluster_count_passed_from_cfg(self):
        cfg = AppConfig()
        cfg.genre_cluster_count = 8
        tracks = sample_tracks(30)

        with patch("generation.genre_mixes.cluster_tracks",
                   return_value=[]) as mock_cluster, \
             patch("generation.genre_mixes.nav_mod"), \
             patch("generation.genre_mixes.notifier"):
            run_genre_mixes(tracks, cfg, MagicMock())

        mock_cluster.assert_called_once_with(tracks, n_clusters=8)
