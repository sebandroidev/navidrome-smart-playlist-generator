"""
Content-based track similarity using beets metadata + optional librosa audio features.
Cosine similarity on (genre_onehot, year_norm, bitrate_norm, bpm_norm[, energy, zcr, spectral_c]).
No GPU required — pure numpy, fast up to ~50K tracks.
"""
import logging
import math
import numpy as np
from collections import Counter

log = logging.getLogger(__name__)

# Genres that are clearly tagging artifacts — excluded from vectors
_JUNK_GENRES = {
    "artist", "various", "unknown", "other", "miscellaneous",
    "1-4 wochen", "1–4 wochen", "neu", "new",
}

_YEAR_MIN, _YEAR_MAX = 1960, 2026
_BITRATE_MIN, _BITRATE_MAX = 64, 1411    # kbps
_BPM_MIN, _BPM_MAX = 40, 220
_SC_MIN, _SC_MAX = 200, 6000             # spectral centroid Hz
_ZCR_MIN, _ZCR_MAX = 0.0, 0.2           # zero crossing rate


def _clean_genre(g: str | None) -> str:
    if not g:
        return ""
    g = g.strip().lower()
    return "" if g in _JUNK_GENRES else g


def _norm(val, lo, hi):
    if val is None:
        return 0.5  # neutral fill for missing values
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def build_feature_matrix(tracks: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Returns:
        matrix    — (N, D) float32 feature matrix, L2-normalised rows
        track_ids — list of track ids aligned with rows
        genres    — list of genre labels aligned with one-hot columns
    """
    # collect top genres (skip junk, keep top 50 by frequency)
    genre_counts: Counter = Counter()
    for t in tracks:
        g = _clean_genre(t.get("genre"))
        if g:
            genre_counts[g] += 1
    top_genres = [g for g, _ in genre_counts.most_common(50)]
    genre_index = {g: i for i, g in enumerate(top_genres)}
    G = len(top_genres)

    track_ids: list[str] = []
    rows: list[list[float]] = []

    for t in tracks:
        tid = t.get("id", "")
        if not tid:
            continue

        # one-hot genre (G dims)
        genre_vec = [0.0] * G
        g = _clean_genre(t.get("genre"))
        if g and g in genre_index:
            genre_vec[genre_index[g]] = 1.0

        # scalar features from beets metadata (4 dims)
        year    = _norm(t.get("year"),    _YEAR_MIN,    _YEAR_MAX)
        bitrate = _norm(t.get("bitrate"), _BITRATE_MIN, _BITRATE_MAX)
        bpm     = _norm(t.get("bpm"),     _BPM_MIN,     _BPM_MAX)
        score   = float(t.get("composite_score") or 0.0)

        # optional librosa audio features (3 dims; 0.5 neutral fill when absent)
        af = t.get("audio_features")
        if isinstance(af, dict) and af:
            energy  = float(af.get("energy", 0.5))
            zcr     = _norm(af.get("zcr"),                _ZCR_MIN, _ZCR_MAX)
            sc      = _norm(af.get("spectral_centroid"),  _SC_MIN,  _SC_MAX)
        else:
            energy, zcr, sc = 0.5, 0.5, 0.5

        row = genre_vec + [year, bitrate, bpm, score, energy, zcr, sc]
        rows.append(row)
        track_ids.append(tid)

    if not rows:
        return np.zeros((0, G + 7), dtype=np.float32), [], top_genres

    matrix = np.array(rows, dtype=np.float32)

    # L2-normalise each row so dot-product == cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    af_count = sum(1 for t in tracks if isinstance(t.get("audio_features"), dict))
    log.info("Similarity: built %dx%d feature matrix (%d genre dims, %d with audio features)",
             matrix.shape[0], matrix.shape[1], G, af_count)
    return matrix, track_ids, top_genres


def top_k_similar(
    seed_ids: list[str],
    all_tracks: list[dict],
    matrix: np.ndarray,
    track_ids: list[str],
    k: int = 50,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """
    Given a list of seed track IDs, return the k most similar tracks
    (by average cosine similarity to seeds) that are NOT in seed_ids or exclude_ids.
    """
    if matrix.shape[0] == 0 or not seed_ids:
        return []

    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    id_to_track = {t["id"]: t for t in all_tracks if t.get("id")}

    seed_indices = [id_to_idx[s] for s in seed_ids if s in id_to_idx]
    if not seed_indices:
        return []

    seed_matrix = matrix[seed_indices]           # (S, D)
    avg_seed = seed_matrix.mean(axis=0)          # (D,)
    avg_seed /= (np.linalg.norm(avg_seed) or 1)  # re-normalise

    scores = matrix @ avg_seed                   # (N,) cosine similarities

    exclude = set(seed_ids) | (exclude_ids or set())
    results = []
    for idx in np.argsort(scores)[::-1]:
        tid = track_ids[idx]
        if tid in exclude:
            continue
        t = id_to_track.get(tid)
        if t and t.get("nav_id"):
            t = dict(t)
            t["similarity_score"] = float(scores[idx])
            results.append(t)
        if len(results) >= k:
            break

    return results
