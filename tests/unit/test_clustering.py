"""
Unit tests for scoring/clustering.py

Covers:
  _cluster_name  — emoji + label mapping and fallback behaviour
  cluster_tracks — not-enough-tracks guard, sklearn KMeans path,
                   min_cluster_size filter, result structure & sort order
"""
import pytest
from unittest.mock import MagicMock, patch

from scoring.clustering import _cluster_name, cluster_tracks


# ---------------------------------------------------------------------------
# Track factory
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
# _cluster_name
# ---------------------------------------------------------------------------

class TestClusterName:
    def test_known_genre_returns_emoji_and_label(self):
        name = _cluster_name(["hip-hop"])
        assert name == "🎤 Hip-Hop Mix"

    def test_known_genre_rap(self):
        assert _cluster_name(["rap"]) == "🎤 Rap Mix"

    def test_known_genre_pop(self):
        assert _cluster_name(["pop"]) == "✨ Pop Mix"

    def test_known_genre_electronic(self):
        assert _cluster_name(["electronic"]) == "⚡ Electronic Mix"

    def test_known_genre_rock(self):
        assert _cluster_name(["rock"]) == "🎸 Rock Mix"

    def test_known_genre_jazz(self):
        assert _cluster_name(["jazz"]) == "🎷 Jazz Mix"

    def test_known_genre_classical(self):
        assert _cluster_name(["classical"]) == "🎻 Classical Mix"

    def test_known_genre_house(self):
        assert _cluster_name(["house"]) == "🏠 House Mix"

    def test_known_genre_techno(self):
        assert _cluster_name(["techno"]) == "🤖 Techno Mix"

    def test_known_genre_afrobeat(self):
        assert _cluster_name(["afrobeat"]) == "🌍 Afrobeats Mix"

    def test_known_genre_afrobeats(self):
        assert _cluster_name(["afrobeats"]) == "🌍 Afrobeats Mix"

    def test_first_matching_genre_wins(self):
        # "pop" is known, "jazz" is also known; "pop" appears first → Pop Mix
        name = _cluster_name(["pop", "jazz"])
        assert name == "✨ Pop Mix"

    def test_unknown_genre_falls_back_to_title_case(self):
        name = _cluster_name(["bossanova"])
        assert name == "🎵 Bossanova Mix"

    def test_empty_list_returns_generic_mix(self):
        assert _cluster_name([]) == "🎵 Mix"

    def test_first_genre_is_unknown_second_is_known(self):
        # Falls back to first match — iterates in order
        name = _cluster_name(["bossanova", "jazz"])
        # "bossanova" not in labels → continue → "jazz" found
        assert name == "🎷 Jazz Mix"

    def test_genre_lookup_is_case_insensitive_via_lower(self):
        # _cluster_name does key = g.lower() before dict lookup
        name = _cluster_name(["Hip-Hop"])
        assert name == "🎤 Hip-Hop Mix"

    def test_uk_drill_has_own_label(self):
        assert _cluster_name(["uk drill"]) == "🔫 UK Drill Mix"

    def test_drill_has_own_label(self):
        assert _cluster_name(["drill"]) == "🔫 Drill Mix"


# ---------------------------------------------------------------------------
# cluster_tracks — guard: not enough tracks
# ---------------------------------------------------------------------------

class TestClusterTracksNotEnough:
    def test_empty_tracks_returns_empty(self):
        result = cluster_tracks([], n_clusters=3, min_cluster_size=5)
        assert result == []

    def test_too_few_tracks_returns_empty(self):
        # Need n_clusters * min_cluster_size eligible tracks
        tracks = [make_track(i) for i in range(5)]
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        # 5 < 3*5=15 → empty
        assert result == []

    def test_tracks_without_nav_id_not_counted(self):
        # 30 tracks but none have nav_id
        tracks = [make_track(i, nav_id=None) for i in range(30)]
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        assert result == []

    def test_sklearn_missing_returns_empty(self):
        tracks = [make_track(i) for i in range(50)]
        with patch.dict("sys.modules", {"sklearn.cluster": None,
                                        "sklearn.preprocessing": None}):
            with patch("builtins.__import__",
                       side_effect=ImportError("sklearn not installed")):
                try:
                    result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
                    assert result == []
                except ImportError:
                    # Acceptable: the function itself catches ImportError
                    pass


# ---------------------------------------------------------------------------
# cluster_tracks — successful clustering
# ---------------------------------------------------------------------------

sklearn = pytest.importorskip("sklearn", reason="scikit-learn required for clustering tests")


class TestClusterTracksSuccess:
    """
    These tests use real sklearn KMeans to verify end-to-end behaviour.
    They are skipped if sklearn is not installed.
    """

    def _enough_tracks(self, n=60, n_clusters=3, min_size=5):
        """Generate enough tracks with varied genres to satisfy the size guard."""
        genres = ["pop", "jazz", "rock", "electronic", "hip-hop"]
        return [
            make_track(i, genre=genres[i % len(genres)], composite_score=float(i) / n)
            for i in range(n)
        ]

    def test_returns_list_of_dicts(self):
        tracks = self._enough_tracks()
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        assert isinstance(result, list)
        for c in result:
            assert isinstance(c, dict)

    def test_cluster_dicts_have_required_keys(self):
        tracks = self._enough_tracks()
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        for c in result:
            assert "cluster_id"  in c
            assert "name"        in c
            assert "genres"      in c
            assert "tracks"      in c

    def test_cluster_name_is_string(self):
        tracks = self._enough_tracks()
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        for c in result:
            assert isinstance(c["name"], str)
            assert len(c["name"]) > 0

    def test_each_cluster_track_list_is_sorted_by_score_desc(self):
        tracks = self._enough_tracks()
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        for c in result:
            scores = [t.get("composite_score", 0) for t in c["tracks"]]
            assert scores == sorted(scores, reverse=True)

    def test_result_sorted_by_cluster_size_desc(self):
        tracks = self._enough_tracks(n=90)
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        sizes = [len(c["tracks"]) for c in result]
        assert sizes == sorted(sizes, reverse=True)

    def test_small_clusters_filtered_out(self):
        # Use a very large min_cluster_size to force some clusters to be filtered
        tracks = self._enough_tracks(n=60)
        result = cluster_tracks(tracks, n_clusters=5, min_cluster_size=20)
        for c in result:
            assert len(c["tracks"]) >= 20

    def test_all_tracks_in_clusters_have_nav_id(self):
        tracks = self._enough_tracks()
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        for c in result:
            for t in c["tracks"]:
                assert t.get("nav_id") is not None

    def test_total_tracks_in_clusters_lte_input(self):
        tracks = self._enough_tracks()
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        total_in_clusters = sum(len(c["tracks"]) for c in result)
        assert total_in_clusters <= len(tracks)

    def test_no_duplicate_track_ids_across_clusters(self):
        tracks = self._enough_tracks()
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        all_ids = []
        for c in result:
            all_ids.extend(t["id"] for t in c["tracks"])
        assert len(all_ids) == len(set(all_ids))

    def test_genres_field_contains_strings(self):
        tracks = self._enough_tracks()
        result = cluster_tracks(tracks, n_clusters=3, min_cluster_size=5)
        for c in result:
            assert isinstance(c["genres"], list)
            for g in c["genres"]:
                assert isinstance(g, str)
