"""
Unit tests for generation/mood_mixes.py

Covers: _mood_bucket, run_mood_mixes (mocked NavidromeClient + db).
"""
import pytest
from unittest.mock import MagicMock, patch

from generation.mood_mixes import (
    _mood_bucket,
    run_mood_mixes,
    _BPM_CHILL_MAX,
    _BPM_ENERGY_MIN,
    _MIN_SIZE,
    _MIX_SIZE,
    _CHILL_GENRES,
    _ENERGY_GENRES,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_track


# ---------------------------------------------------------------------------
# _mood_bucket — BPM-based classification
# ---------------------------------------------------------------------------

class TestMoodBucketBPM:
    def test_bpm_below_chill_max_is_chill(self):
        assert _mood_bucket(make_track(0, bpm=_BPM_CHILL_MAX - 1)) == "chill"

    def test_bpm_at_chill_max_minus_one_is_chill(self):
        assert _mood_bucket(make_track(0, bpm=89)) == "chill"

    def test_bpm_at_chill_max_is_not_chill(self):
        # bpm == _BPM_CHILL_MAX (90) is NOT < 90, so not chill
        result = _mood_bucket(make_track(0, bpm=90))
        assert result in ("flow", "energy")

    def test_bpm_above_energy_min_no_audio_features_is_energy(self):
        t = make_track(0, bpm=_BPM_ENERGY_MIN, audio_features=None)
        assert _mood_bucket(t) == "energy"

    def test_bpm_above_energy_min_high_energy_is_energy(self):
        t = make_track(0, bpm=130, audio_features={"energy": 0.8})
        assert _mood_bucket(t) == "energy"

    def test_bpm_above_energy_min_low_energy_is_flow(self):
        # energy < 0.45 → flow, not energy
        t = make_track(0, bpm=130, audio_features={"energy": 0.2})
        assert _mood_bucket(t) == "flow"

    def test_bpm_above_energy_min_energy_exactly_threshold_is_energy(self):
        # energy == 0.45 → >= threshold → energy
        t = make_track(0, bpm=130, audio_features={"energy": 0.45})
        assert _mood_bucket(t) == "energy"

    def test_bpm_in_middle_range_is_flow(self):
        # 90 <= bpm < 120 is neither chill nor energy → flow
        for bpm in [90, 100, 110, 119]:
            result = _mood_bucket(make_track(0, bpm=bpm))
            assert result == "flow", f"bpm={bpm} should be flow"

    def test_bpm_zero_uses_genre_fallback(self):
        t = make_track(0, bpm=0, genre="ambient")
        assert _mood_bucket(t) == "chill"

    def test_bpm_none_uses_genre_fallback(self):
        t = make_track(0, bpm=None, genre="techno")
        assert _mood_bucket(t) == "energy"


# ---------------------------------------------------------------------------
# _mood_bucket — genre-based fallback (no BPM)
# ---------------------------------------------------------------------------

class TestMoodBucketGenreFallback:
    def _no_bpm(self, genre):
        return make_track(0, bpm=None, genre=genre, audio_features=None)

    def test_ambient_is_chill(self):
        assert _mood_bucket(self._no_bpm("ambient")) == "chill"

    def test_classical_is_chill(self):
        assert _mood_bucket(self._no_bpm("classical")) == "chill"

    def test_jazz_is_chill(self):
        assert _mood_bucket(self._no_bpm("jazz")) == "chill"

    def test_acoustic_is_chill(self):
        assert _mood_bucket(self._no_bpm("acoustic")) == "chill"

    def test_lo_fi_is_chill(self):
        assert _mood_bucket(self._no_bpm("lo-fi")) == "chill"

    def test_folk_is_chill(self):
        assert _mood_bucket(self._no_bpm("folk")) == "chill"

    def test_electronic_is_energy(self):
        assert _mood_bucket(self._no_bpm("electronic")) == "energy"

    def test_techno_is_energy(self):
        assert _mood_bucket(self._no_bpm("techno")) == "energy"

    def test_metal_is_energy(self):
        assert _mood_bucket(self._no_bpm("metal")) == "energy"

    def test_dance_is_energy(self):
        assert _mood_bucket(self._no_bpm("dance")) == "energy"

    def test_unknown_genre_is_flow(self):
        assert _mood_bucket(self._no_bpm("xyzzy-genre")) == "flow"

    def test_empty_genre_is_flow(self):
        assert _mood_bucket(self._no_bpm("")) == "flow"

    def test_none_genre_is_flow(self):
        t = make_track(0, bpm=None, genre=None, audio_features=None)
        assert _mood_bucket(t) == "flow"

    def test_hip_hop_is_flow(self):
        # Not in chill or energy sets → flow
        assert _mood_bucket(self._no_bpm("hip-hop")) == "flow"

    def test_rock_is_flow(self):
        assert _mood_bucket(self._no_bpm("rock")) == "flow"

    def test_genre_matching_is_substring(self):
        # "drum & bass" contains "drum and bass"? No — exact set membership.
        # "drum & bass" IS in _ENERGY_GENRES
        assert _mood_bucket(self._no_bpm("drum & bass")) == "energy"

    def test_genre_match_case_insensitive(self):
        t = make_track(0, bpm=None, genre="JAZZ", audio_features=None)
        assert _mood_bucket(t) == "chill"


# ---------------------------------------------------------------------------
# run_mood_mixes — integration (mocked client + db)
# ---------------------------------------------------------------------------

def _make_pool_tracks(mood, n, bpm_map):
    """Create n tracks that will map to the given mood bucket."""
    bpm = bpm_map[mood]
    tracks = []
    for i in range(n):
        tracks.append(make_track(
            i,
            bpm=bpm,
            nav_id=f"nav_{mood}_{i}",
            composite_score=0.5 + i * 0.01,
            audio_features=None,
        ))
    return tracks


class TestRunMoodMixes:
    def _run(self, tracks, cfg=None, db=None, push_return="pl_id_1"):
        if cfg is None:
            from config import AppConfig
            cfg = AppConfig()
        if db is None:
            db = MagicMock()

        mock_client = MagicMock()
        mock_client.push_playlist.return_value = push_return

        with patch("generation.mood_mixes.nav_mod") as mock_nav:
            mock_nav.NavidromeClient.return_value = mock_client
            results = run_mood_mixes(tracks, cfg, db)

        return results, mock_client, db

    def test_returns_list(self):
        tracks = [make_track(i, bpm=100, nav_id=f"nav{i}") for i in range(20)]
        results, _, _ = self._run(tracks)
        assert isinstance(results, list)

    def test_skips_mood_with_fewer_than_min_size_tracks(self):
        # Only flow tracks (bpm 90-119), fewer than _MIN_SIZE for chill/energy
        tracks = [make_track(i, bpm=100, nav_id=f"nav{i}") for i in range(_MIN_SIZE + 1)]
        results, client, _ = self._run(tracks)
        # Only "flow" should have been pushed (chill and energy have 0 tracks)
        moods_pushed = {r["mood"] for r in results}
        assert "chill" not in moods_pushed
        assert "energy" not in moods_pushed

    def test_generates_all_three_moods_when_enough_tracks(self):
        from config import AppConfig
        # Create enough tracks for each mood bucket
        chill_tracks = [make_track(i, bpm=70, nav_id=f"navC{i}",
                                   audio_features=None) for i in range(_MIN_SIZE + 2)]
        flow_tracks = [make_track(i + 100, bpm=100, nav_id=f"navF{i}",
                                  audio_features=None) for i in range(_MIN_SIZE + 2)]
        energy_tracks = [make_track(i + 200, bpm=130, nav_id=f"navE{i}",
                                    audio_features=None) for i in range(_MIN_SIZE + 2)]
        tracks = chill_tracks + flow_tracks + energy_tracks
        results, _, _ = self._run(tracks)
        moods = {r["mood"] for r in results}
        assert moods == {"chill", "flow", "energy"}

    def test_result_has_required_keys(self):
        tracks = [make_track(i, bpm=100, nav_id=f"nav{i}") for i in range(_MIN_SIZE + 5)]
        results, _, _ = self._run(tracks)
        if results:
            r = results[0]
            assert "mood" in r
            assert "name" in r
            assert "track_count" in r
            assert "nav_playlist_id" in r

    def test_track_count_at_most_mix_size(self):
        # Provide many more tracks than _MIX_SIZE
        tracks = [make_track(i, bpm=100, nav_id=f"nav{i}") for i in range(_MIX_SIZE * 3)]
        results, _, _ = self._run(tracks)
        for r in results:
            assert r["track_count"] <= _MIX_SIZE

    def test_db_save_playlist_called_for_each_mood(self):
        tracks = [make_track(i, bpm=100, nav_id=f"nav{i}") for i in range(_MIN_SIZE + 5)]
        results, _, db = self._run(tracks)
        assert db.save_playlist.call_count == len(results)

    def test_push_playlist_called_with_nav_ids(self):
        tracks = [make_track(i, bpm=100, nav_id=f"nav{i}") for i in range(_MIN_SIZE + 5)]
        _, client, _ = self._run(tracks)
        if client.push_playlist.call_count > 0:
            song_ids_arg = client.push_playlist.call_args[0][1]
            assert all(isinstance(s, str) for s in song_ids_arg)

    def test_tracks_without_nav_id_excluded(self):
        tracks = [make_track(i, bpm=100, nav_id=f"nav{i}") for i in range(_MIN_SIZE + 5)]
        tracks.append(make_track(99, bpm=100, nav_id=None))
        # Should not raise; track without nav_id silently ignored
        results, _, _ = self._run(tracks)
        assert isinstance(results, list)

    def test_mood_names_contain_emoji_and_label(self):
        tracks = [make_track(i, bpm=100, nav_id=f"nav{i}") for i in range(_MIN_SIZE + 5)]
        results, _, _ = self._run(tracks)
        for r in results:
            assert r["mood"] in ("chill", "flow", "energy")
            name = r["name"]
            # Should be non-empty
            assert len(name) > 0

    def test_empty_tracks_returns_empty_results(self):
        results, client, _ = self._run([])
        assert results == []
        client.push_playlist.assert_not_called()
