"""
Genre cluster playlist generator.
Runs K-means on track feature vectors → N named playlists pushed to Navidrome.
"""
import logging
from config import AppConfig
from state.db import StateDB
from ingestion import navidrome as nav_mod
from scoring.clustering import cluster_tracks
from generation.constraints import deduplicate, filter_unmatched, fix_consecutive_artists
import notifier

log = logging.getLogger(__name__)

# How many tracks per genre mix playlist
_MIX_SIZE = 40


def run_genre_mixes(tracks: list[dict], cfg: AppConfig, db: StateDB) -> list[dict]:
    """
    Cluster tracks, generate one playlist per cluster in Navidrome.
    Returns list of result dicts { name, cluster_id, track_count, nav_playlist_id }.
    """
    n_clusters = getattr(cfg, "genre_cluster_count", 5)
    clusters = cluster_tracks(tracks, n_clusters=n_clusters)

    if not clusters:
        log.warning("Genre mixes: no clusters produced (scikit-learn installed?)")
        return []

    client = nav_mod.NavidromeClient(
        cfg.navidrome.url, cfg.navidrome.user, cfg.navidrome.password
    )

    results = []
    names_used: set[str] = set()

    for c in clusters:
        # De-duplicate playlist name if two clusters share a label
        name = c["name"]
        if name in names_used:
            name = f"{name} II"
        names_used.add(name)

        # Take top-scored tracks from cluster, apply constraints
        picks = c["tracks"][:_MIX_SIZE * 2]  # fetch extra for constraint headroom
        picks = deduplicate(picks)
        picks = filter_unmatched(picks)
        picks = fix_consecutive_artists(picks)
        picks = picks[:_MIX_SIZE]

        if not picks:
            continue

        song_ids = [t["nav_id"] for t in picks]
        pl_id = client.push_playlist(name, song_ids)

        db.save_playlist("genre_mix", [t["id"] for t in picks], pl_id)

        log.info("Genre mix '%s' → %d tracks (playlist %s)", name, len(picks), pl_id)
        results.append({
            "name":            name,
            "cluster_id":      c["cluster_id"],
            "genres":          c["genres"],
            "track_count":     len(picks),
            "nav_playlist_id": pl_id,
        })

    if results:
        names = ", ".join(r["name"] for r in results)
        notifier.notify(
            f"🎛 Genre mixes refreshed — {len(results)} playlists\n_{names}_",
            cfg,
        )

    return results
