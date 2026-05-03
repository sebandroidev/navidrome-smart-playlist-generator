"""
Unit tests for scoring/similarity.py

Covers:
  _clean_genre, _norm, build_feature_matrix, top_k_similar
"""
import math
import numpy as np
import pytest

from scoring.similarity import (
    _clean_genre,
    _norm,
    build_feature_matrix,
    top_k_similar,
    _JUNK_GENRES,
    _YEAR_MIN, _YEAR_MAX,
    _BITRATE_MIN, _BITRATE_MAX,
    _BPM_MIN, _BPM_MAX,
    _ZCR_MIN, _ZCR_MAX,
    _SC_MIN, _SC_MAX,
)


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
# _clean_genre
# ---------------------------------------------------------------------------

class TestCleanGenre:
    def test_none_returns_empty_string(self):
        assert _clean_genre(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert _clean_genre("") == ""

    def test_normal_genre_lowercased_and_stripped(self):
        assert _clean_genre("  Hip-Hop  ") == "hip-hop"

    def test_junk_genres_return_empty_string(self):
        for junk in _JUNK_GENRES:
            assert _clean_genre(junk) == "", f"Expected empty for junk genre '{junk}'"

    def test_junk_genre_case_insensitive(self):
        assert _clean_genre("Artist") == ""
        assert _clean_genre("UNKNOWN") == ""

    def test_valid_genre_kept(self):
        assert _clean_genre("Jazz") == "jazz"
        assert _clean_genre("R&B") == "r&b"


# ---------------------------------------------------------------------------
# _norm
# ---------------------------------------------------------------------------

class TestNorm:
    def test_none_returns_neutral_fill(self):
        assert _norm(None, 0, 100) == pytest.approx(0.5)

    def test_at_min_returns_zero(self):
        assert _norm(0, 0, 100) == pytest.approx(0.0)

    def test_at_max_returns_one(self):
        assert _norm(100, 0, 100) == pytest.approx(1.0)

    def test_midpoint_returns_half(self):
        assert _norm(50, 0, 100) == pytest.approx(0.5)

    def test_below_min_clamped_to_zero(self):
        assert _norm(-10, 0, 100) == pytest.approx(0.0)

    def test_above_max_clamped_to_one(self):
        assert _norm(200, 0, 100) == pytest.approx(1.0)

    def test_year_min_boundary(self):
        assert _norm(_YEAR_MIN, _YEAR_MIN, _YEAR_MAX) == pytest.approx(0.0)

    def test_year_max_boundary(self):
        assert _norm(_YEAR_MAX, _YEAR_MIN, _YEAR_MAX) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_feature_matrix — shape and structure
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrixShape:
    def test_empty_tracks_returns_zero_rows(self):
        matrix, track_ids, genres = build_feature_matrix([])
        assert matrix.shape[0] == 0
        assert track_ids == []

    def test_empty_tracks_zero_matrix_columns(self):
        # G == 0 genres, + 7 scalar dims → shape (0, 7)
        matrix, _, genres = build_feature_matrix([])
        assert matrix.shape == (0, len(genres) + 7)

    def test_single_track_produces_one_row(self):
        matrix, track_ids, genres = build_feature_matrix([make_track(0)])
        assert matrix.shape[0] == 1
        assert len(track_ids) == 1

    def test_n_tracks_produces_n_rows(self):
        tracks = [make_track(i) for i in range(10)]
        matrix, track_ids, _ = build_feature_matrix(tracks)
        assert matrix.shape[0] == 10
        assert len(track_ids) == 10

    def test_column_count_is_genres_plus_seven(self):
        tracks = [make_track(i, genre=f"genre{i}") for i in range(5)]
        matrix, _, genres = build_feature_matrix(tracks)
        assert matrix.shape[1] == len(genres) + 7

    def test_dtype_is_float32(self):
        tracks = [make_track(0)]
        matrix, _, _ = build_feature_matrix(tracks)
        assert matrix.dtype == np.float32

    def test_track_without_id_excluded(self):
        tracks = [make_track(0), {"title": "No ID", "genre": "pop"}]
        matrix, track_ids, _ = build_feature_matrix(tracks)
        assert matrix.shape[0] == 1
        assert all(tid != "" for tid in track_ids)

    def test_track_ids_align_with_rows(self):
        tracks = [make_track(i) for i in range(5)]
        matrix, track_ids, _ = build_feature_matrix(tracks)
        for i, tid in enumerate(track_ids):
            assert tid == tracks[i]["id"]


# ---------------------------------------------------------------------------
# build_feature_matrix — normalisation
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrixNormalization:
    def test_rows_are_l2_unit_vectors(self):
        tracks = [make_track(i, genre=f"g{i % 3}") for i in range(5)]
        matrix, _, _ = build_feature_matrix(tracks)
        norms = np.linalg.norm(matrix, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)

    def test_zero_vector_row_not_divided_by_zero(self):
        # A track where all features are 0 — genre junk, all metadata None/0
        t = {
            "id": "zero-track",
            "genre": "artist",  # junk genre → zero one-hot
            "year": None, "bitrate": None, "bpm": None,
            "composite_score": 0.0,
            "audio_features": {"energy": 0.5, "zcr": 0.0, "spectral_centroid": 0.0},
        }
        # Should not raise; row norm replaced with 1
        matrix, _, _ = build_feature_matrix([t])
        assert not np.any(np.isnan(matrix))

    def test_top_genres_capped_at_50(self):
        # 60 unique genres → only top 50 kept
        tracks = [make_track(i, genre=f"genre_{i}") for i in range(60)]
        _, _, genres = build_feature_matrix(tracks)
        assert len(genres) <= 50


# ---------------------------------------------------------------------------
# build_feature_matrix — audio features
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrixAudioFeatures:
    def test_absent_audio_features_filled_with_neutral(self):
        t = make_track(0, audio_features=None)
        matrix, _, genres = build_feature_matrix([t])
        G = len(genres)
        # energy, zcr, sc are at positions G+4, G+5, G+6
        # All three start at the same neutral 0.5 fill, so after L2-norm they're equal
        e, z, sc = matrix[0, G + 4], matrix[0, G + 5], matrix[0, G + 6]
        assert e > 0 and z > 0 and sc > 0
        assert e == pytest.approx(z, abs=1e-5) and z == pytest.approx(sc, abs=1e-5)

    def test_audio_features_dict_used_when_present(self):
        # Two identical tracks except one has extreme audio features
        t_base  = make_track(0, genre="pop", audio_features=None)
        t_audio = make_track(1, genre="pop",
                             audio_features={"energy": 1.0, "zcr": 0.2, "spectral_centroid": 6000})
        matrix, track_ids, genres = build_feature_matrix([t_base, t_audio])
        G = len(genres)
        idx_base  = track_ids.index(t_base["id"])
        idx_audio = track_ids.index(t_audio["id"])
        # The raw (pre-norm) energy dim is G+4; after L2-norm values differ
        # We just verify the rows are not identical
        assert not np.allclose(matrix[idx_base], matrix[idx_audio])

    def test_none_audio_features_dict_falls_back_to_neutral(self):
        t = make_track(0, audio_features={})
        matrix, _, genres = build_feature_matrix([t])
        G = len(genres)
        # Empty dict → falls back to neutral fill
        # After L2-norm we can't check exact value, but row should be valid
        assert not np.any(np.isnan(matrix[0]))

    def test_partial_audio_features_zcr_none_neutral(self):
        t = make_track(0, audio_features={"energy": 0.8})
        # zcr not present → _norm(None, …) → 0.5
        matrix, _, genres = build_feature_matrix([t])
        assert not np.any(np.isnan(matrix[0]))


# ---------------------------------------------------------------------------
# top_k_similar
# ---------------------------------------------------------------------------

class TestTopKSimilar:
    def _setup(self, n=10, same_genre="pop"):
        tracks = [make_track(i, genre=same_genre) for i in range(n)]
        matrix, track_ids, _ = build_feature_matrix(tracks)
        return tracks, matrix, track_ids

    def test_empty_matrix_returns_empty(self):
        matrix = np.zeros((0, 7), dtype=np.float32)
        result = top_k_similar(["id0"], [], matrix, [], k=5)
        assert result == []

    def test_empty_seed_ids_returns_empty(self):
        tracks, matrix, track_ids = self._setup()
        result = top_k_similar([], tracks, matrix, track_ids, k=5)
        assert result == []

    def test_seed_not_in_matrix_returns_empty(self):
        tracks, matrix, track_ids = self._setup()
        result = top_k_similar(["nonexistent-id"], tracks, matrix, track_ids, k=5)
        assert result == []

    def test_seeds_excluded_from_results(self):
        tracks, matrix, track_ids = self._setup(n=10)
        seed_ids = [tracks[0]["id"]]
        result = top_k_similar(seed_ids, tracks, matrix, track_ids, k=9)
        result_ids = {t["id"] for t in result}
        assert tracks[0]["id"] not in result_ids

    def test_exclude_ids_not_in_results(self):
        tracks, matrix, track_ids = self._setup(n=10)
        seed_ids    = [tracks[0]["id"]]
        exclude_ids = {tracks[1]["id"], tracks[2]["id"]}
        result = top_k_similar(seed_ids, tracks, matrix, track_ids,
                               k=10, exclude_ids=exclude_ids)
        result_ids = {t["id"] for t in result}
        assert result_ids.isdisjoint(exclude_ids)

    def test_returns_at_most_k_results(self):
        tracks, matrix, track_ids = self._setup(n=20)
        seed_ids = [tracks[0]["id"]]
        result = top_k_similar(seed_ids, tracks, matrix, track_ids, k=5)
        assert len(result) <= 5

    def test_similarity_score_attached_to_results(self):
        tracks, matrix, track_ids = self._setup(n=10)
        seed_ids = [tracks[0]["id"]]
        result = top_k_similar(seed_ids, tracks, matrix, track_ids, k=3)
        for t in result:
            assert "similarity_score" in t
            assert isinstance(t["similarity_score"], float)

    def test_tracks_without_nav_id_excluded_from_results(self):
        tracks = [make_track(i) for i in range(10)]
        # Strip nav_id from half of them
        for t in tracks[5:]:
            t["nav_id"] = None
        matrix, track_ids, _ = build_feature_matrix(tracks)
        seed_ids = [tracks[0]["id"]]
        result = top_k_similar(seed_ids, tracks, matrix, track_ids, k=10)
        for t in result:
            assert t.get("nav_id") is not None

    def test_similar_genre_track_scores_higher_than_different_genre(self):
        # Seed is "jazz"; make one similar jazz track and one very different pop/electronic track
        seed = make_track(0, genre="jazz", year=1965, bpm=80, bitrate=192,
                          composite_score=0.5)
        similar = make_track(1, genre="jazz", year=1968, bpm=85, bitrate=192,
                             composite_score=0.5)
        different = make_track(2, genre="electronic", year=2022, bpm=140, bitrate=320,
                               composite_score=0.5)

        tracks = [seed, similar, different]
        matrix, track_ids, _ = build_feature_matrix(tracks)
        result = top_k_similar([seed["id"]], tracks, matrix, track_ids, k=2)

        assert len(result) == 2
        # The jazz track should rank first (higher similarity_score)
        assert result[0]["id"] == similar["id"]

    def test_result_tracks_are_copies_not_originals(self):
        # top_k_similar creates dict copies; mutating result should not affect originals
        tracks, matrix, track_ids = self._setup(n=5)
        seed_ids = [tracks[0]["id"]]
        result = top_k_similar(seed_ids, tracks, matrix, track_ids, k=2)
        if result:
            original_title = tracks[1]["title"]
            result[0]["title"] = "MUTATED"
            assert tracks[1]["title"] == original_title
