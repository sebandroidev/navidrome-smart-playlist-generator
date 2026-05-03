"""
K-means genre clustering.
Groups tracks into taste clusters; each cluster becomes a named playlist.
"""
import logging
import numpy as np
from collections import Counter
from scoring.similarity import build_feature_matrix, _clean_genre

log = logging.getLogger(__name__)

# Genre → emoji + display label overrides
_GENRE_LABELS = {
    "rap":          ("🎤", "Rap"),
    "hip hop":      ("🎤", "Hip-Hop"),
    "hip-hop":      ("🎤", "Hip-Hop"),
    "uk drill":     ("🔫", "UK Drill"),
    "drill":        ("🔫", "Drill"),
    "afrobeat":     ("🌍", "Afrobeats"),
    "afrobeats":    ("🌍", "Afrobeats"),
    "pop":          ("✨", "Pop"),
    "electronic":   ("⚡", "Electronic"),
    "r&b":          ("🎸", "R&B"),
    "rnb":          ("🎸", "R&B"),
    "soul":         ("🎷", "Soul"),
    "jazz":         ("🎷", "Jazz"),
    "rock":         ("🎸", "Rock"),
    "indie":        ("🎸", "Indie"),
    "classical":    ("🎻", "Classical"),
    "j-pop":        ("🌸", "J-Pop"),
    "k-pop":        ("🌸", "K-Pop"),
    "reggae":       ("🌴", "Reggae"),
    "dancehall":    ("🌴", "Dancehall"),
    "house":        ("🏠", "House"),
    "techno":       ("🤖", "Techno"),
    "trap":         ("🔊", "Trap"),
    "metal":        ("🤘", "Metal"),
    "punk":         ("🤘", "Punk"),
    "country":      ("🤠", "Country"),
    "folk":         ("🪕", "Folk"),
    "latin":        ("💃", "Latin"),
    "blues":        ("🎵", "Blues"),
    "gospel":       ("🙏", "Gospel"),
}


def _cluster_name(dominant_genres: list[str]) -> str:
    for g in dominant_genres:
        key = g.lower()
        if key in _GENRE_LABELS:
            emoji, label = _GENRE_LABELS[key]
            return f"{emoji} {label} Mix"
    # fallback: capitalise first genre
    if dominant_genres:
        return f"🎵 {dominant_genres[0].title()} Mix"
    return "🎵 Mix"


def cluster_tracks(
    tracks: list[dict],
    n_clusters: int = 5,
    min_cluster_size: int = 10,
) -> list[dict]:
    """
    Returns list of cluster dicts:
      { name, cluster_id, tracks: [track_dict, ...] }
    Sorted by cluster size descending.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
    except ImportError:
        log.warning("scikit-learn not installed; genre clustering unavailable")
        return []

    eligible = [t for t in tracks if t.get("nav_id")]
    if len(eligible) < n_clusters * min_cluster_size:
        log.warning("Not enough tracks (%d) for %d clusters", len(eligible), n_clusters)
        return []

    matrix, track_ids, genre_cols = build_feature_matrix(eligible)
    if matrix.shape[0] == 0:
        return []

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)

    id_to_track = {t["id"]: t for t in eligible if t.get("id")}
    clusters: dict[int, list[dict]] = {}
    for idx, label in enumerate(labels):
        tid = track_ids[idx]
        t = id_to_track.get(tid)
        if t:
            clusters.setdefault(int(label), []).append(t)

    result = []
    for cid, ctracks in clusters.items():
        if len(ctracks) < min_cluster_size:
            continue

        # find dominant genres for naming
        genre_counts: Counter = Counter()
        for t in ctracks:
            g = _clean_genre(t.get("genre"))
            if g:
                genre_counts[g] += 1
        dominant = [g for g, _ in genre_counts.most_common(3)]

        result.append({
            "cluster_id": cid,
            "name":       _cluster_name(dominant),
            "genres":     dominant,
            "tracks":     sorted(ctracks,
                                 key=lambda t: t.get("composite_score", 0),
                                 reverse=True),
        })

    result.sort(key=lambda c: len(c["tracks"]), reverse=True)
    log.info("Clustering: %d clusters from %d tracks", len(result), len(eligible))
    return result
