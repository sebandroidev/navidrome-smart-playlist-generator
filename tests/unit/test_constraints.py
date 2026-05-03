"""
Unit tests for generation/constraints.py

Covers:
  deduplicate          — removes duplicate track ids, preserves first occurrence
  filter_unmatched     — drops tracks without nav_id
  fix_consecutive_artists — greedy swap eliminates back-to-back same-artist runs
  apply_all            — chains all three in correct order
"""
import pytest

from generation.constraints import (
    deduplicate,
    filter_unmatched,
    fix_consecutive_artists,
    apply_all,
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
# deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_empty_list_returns_empty(self):
        assert deduplicate([]) == []

    def test_no_duplicates_unchanged(self):
        tracks = [make_track(i) for i in range(5)]
        result = deduplicate(tracks)
        assert len(result) == 5

    def test_duplicates_removed(self):
        t = make_track(0)
        result = deduplicate([t, t, t])
        assert len(result) == 1

    def test_first_occurrence_kept(self):
        t1 = make_track(0)
        t2 = dict(t1)  # same id, different object
        t2["title"] = "Second"
        result = deduplicate([t1, t2])
        assert result[0]["title"] == t1["title"]

    def test_tracks_without_id_excluded(self):
        # Tracks with empty string id are silently dropped (id is falsy)
        t_no_id = {"id": "", "nav_id": "nav0", "title": "No ID"}
        t_valid = make_track(1)
        result = deduplicate([t_no_id, t_valid])
        assert len(result) == 1
        assert result[0]["id"] == t_valid["id"]

    def test_tracks_with_none_id_excluded(self):
        t_none = make_track(0)
        t_none["id"] = None
        t_valid = make_track(1)
        result = deduplicate([t_none, t_valid])
        assert len(result) == 1

    def test_order_preserved(self):
        tracks = [make_track(i) for i in range(5, 0, -1)]
        result = deduplicate(tracks)
        assert [t["id"] for t in result] == [t["id"] for t in tracks]

    def test_mixed_duplicates_and_unique_tracks(self):
        tracks = [
            make_track(0),
            make_track(1),
            make_track(0),  # duplicate of first
            make_track(2),
            make_track(1),  # duplicate of second
        ]
        result = deduplicate(tracks)
        assert len(result) == 3
        result_ids = [t["id"] for t in result]
        assert result_ids == [make_track(0)["id"],
                              make_track(1)["id"],
                              make_track(2)["id"]]


# ---------------------------------------------------------------------------
# filter_unmatched
# ---------------------------------------------------------------------------

class TestFilterUnmatched:
    def test_empty_list_returns_empty(self):
        assert filter_unmatched([]) == []

    def test_all_matched_returns_all(self):
        tracks = [make_track(i) for i in range(5)]
        result = filter_unmatched(tracks)
        assert len(result) == 5

    def test_tracks_without_nav_id_removed(self):
        tracks = [make_track(i) for i in range(4)]
        tracks[1]["nav_id"] = None
        tracks[3]["nav_id"] = None
        result = filter_unmatched(tracks)
        assert len(result) == 2
        for t in result:
            assert t.get("nav_id") is not None

    def test_empty_string_nav_id_treated_as_missing(self):
        t = make_track(0, nav_id="")
        result = filter_unmatched([t])
        # Empty string is falsy → dropped
        assert result == []

    def test_valid_nav_ids_preserved(self):
        t = make_track(0, nav_id="abc123")
        result = filter_unmatched([t])
        assert result[0]["nav_id"] == "abc123"

    def test_all_unmatched_returns_empty(self):
        tracks = [make_track(i, nav_id=None) for i in range(3)]
        result = filter_unmatched(tracks)
        assert result == []

    def test_order_preserved(self):
        tracks = [make_track(i) for i in range(5)]
        tracks[2]["nav_id"] = None
        result = filter_unmatched(tracks)
        expected_ids = [tracks[0]["id"], tracks[1]["id"],
                        tracks[3]["id"], tracks[4]["id"]]
        assert [t["id"] for t in result] == expected_ids


# ---------------------------------------------------------------------------
# fix_consecutive_artists
# ---------------------------------------------------------------------------

class TestFixConsecutiveArtists:
    def test_empty_list_returned_unchanged(self):
        assert fix_consecutive_artists([]) == []

    def test_single_track_returned_unchanged(self):
        t = make_track(0)
        result = fix_consecutive_artists([t])
        assert result == [t]

    def test_no_consecutive_artists_unchanged(self):
        tracks = [
            make_track(0, artist="Alice"),
            make_track(1, artist="Bob"),
            make_track(2, artist="Charlie"),
        ]
        original_ids = [t["id"] for t in tracks]
        result = fix_consecutive_artists(list(tracks))
        assert [t["id"] for t in result] == original_ids

    def test_consecutive_pair_resolved(self):
        tracks = [
            make_track(0, artist="Alice"),
            make_track(1, artist="Alice"),
            make_track(2, artist="Bob"),
        ]
        result = fix_consecutive_artists(list(tracks))
        # No two adjacent tracks should share the same artist
        for i in range(len(result) - 1):
            a1 = (result[i].get("artist") or "").lower()
            a2 = (result[i + 1].get("artist") or "").lower()
            assert a1 != a2, f"Consecutive artist '{a1}' at position {i}"

    def test_multiple_consecutive_runs_resolved(self):
        tracks = [
            make_track(0, artist="Alice"),
            make_track(1, artist="Alice"),
            make_track(2, artist="Bob"),
            make_track(3, artist="Bob"),
            make_track(4, artist="Charlie"),
        ]
        result = fix_consecutive_artists(list(tracks))
        for i in range(len(result) - 1):
            a1 = (result[i].get("artist") or "").lower()
            a2 = (result[i + 1].get("artist") or "").lower()
            assert a1 != a2, f"Consecutive artist '{a1}' at position {i}"

    def test_same_number_of_tracks_after_fix(self):
        tracks = [
            make_track(0, artist="Alice"),
            make_track(1, artist="Alice"),
            make_track(2, artist="Bob"),
        ]
        result = fix_consecutive_artists(list(tracks))
        assert len(result) == 3

    def test_all_same_artist_cannot_be_fixed(self):
        # When every track is the same artist, no swap is possible;
        # function should return without infinite loop
        tracks = [make_track(i, artist="Alice") for i in range(4)]
        result = fix_consecutive_artists(list(tracks))
        assert len(result) == 4  # unchanged length; no crash

    def test_artist_comparison_is_case_insensitive(self):
        tracks = [
            make_track(0, artist="alice"),
            make_track(1, artist="ALICE"),  # same artist, different case
            make_track(2, artist="Bob"),
        ]
        result = fix_consecutive_artists(list(tracks))
        # Should treat "alice" == "ALICE" and attempt a swap
        for i in range(len(result) - 1):
            a1 = (result[i].get("artist") or "").lower()
            a2 = (result[i + 1].get("artist") or "").lower()
            assert a1 != a2

    def test_max_passes_limits_iterations(self):
        # Just verify max_passes param is accepted without error
        tracks = [make_track(i, artist="X" if i % 2 == 0 else "Y") for i in range(6)]
        result = fix_consecutive_artists(list(tracks), max_passes=1)
        assert len(result) == 6

    def test_returns_list_not_none(self):
        tracks = [make_track(i) for i in range(3)]
        result = fix_consecutive_artists(tracks)
        assert result is not None
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# apply_all
# ---------------------------------------------------------------------------

class TestApplyAll:
    def test_empty_list_returns_empty(self):
        assert apply_all([]) == []

    def test_duplicates_removed(self):
        t = make_track(0)
        result = apply_all([t, t])
        assert len(result) == 1

    def test_unmatched_removed(self):
        tracks = [
            make_track(0, nav_id="valid"),
            make_track(1, nav_id=None),
        ]
        result = apply_all(tracks)
        assert len(result) == 1
        assert result[0]["nav_id"] == "valid"

    def test_consecutive_artists_fixed(self):
        tracks = [
            make_track(0, artist="Alice", nav_id="n0"),
            make_track(1, artist="Alice", nav_id="n1"),
            make_track(2, artist="Bob",   nav_id="n2"),
        ]
        result = apply_all(list(tracks))
        for i in range(len(result) - 1):
            a1 = (result[i].get("artist") or "").lower()
            a2 = (result[i + 1].get("artist") or "").lower()
            assert a1 != a2

    def test_pipeline_applies_all_three_transforms(self):
        # Combine: duplicate + unmatched + consecutive artist issue
        dup = make_track(0, artist="Alice", nav_id="n0")
        tracks = [
            dup,
            dup,                                           # duplicate
            make_track(1, artist="Alice", nav_id=None),    # unmatched
            make_track(2, artist="Alice", nav_id="n2"),    # consecutive Alice
            make_track(3, artist="Bob",   nav_id="n3"),
        ]
        result = apply_all(list(tracks))
        # Duplicate removed → 4 remaining; unmatched removed → 3 remaining
        assert len(result) == 3
        # No two adjacent same artist
        for i in range(len(result) - 1):
            a1 = (result[i].get("artist") or "").lower()
            a2 = (result[i + 1].get("artist") or "").lower()
            assert a1 != a2

    def test_all_valid_unique_tracks_preserved(self):
        tracks = [make_track(i) for i in range(5)]
        result = apply_all(list(tracks))
        # All have unique ids and nav_ids; only consecutive-artist swap may reorder
        assert len(result) == 5
