import logging

log = logging.getLogger(__name__)


def deduplicate(tracks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for t in tracks:
        tid = t.get("id", "")
        if tid and tid not in seen:
            seen.add(tid)
            out.append(t)
    return out


def fix_consecutive_artists(tracks: list[dict], max_passes: int = 3) -> list[dict]:
    """Greedy swap to eliminate back-to-back same-artist runs."""
    if len(tracks) < 2:
        return tracks

    for _ in range(max_passes):
        changed = False
        for i in range(len(tracks) - 1):
            a1 = (tracks[i].get("artist") or "").lower()
            a2 = (tracks[i + 1].get("artist") or "").lower()
            if a1 == a2:
                # find the next track with a different artist to swap in
                for j in range(i + 2, len(tracks)):
                    if (tracks[j].get("artist") or "").lower() != a1:
                        tracks[i + 1], tracks[j] = tracks[j], tracks[i + 1]
                        changed = True
                        break
        if not changed:
            break

    return tracks


def filter_unmatched(tracks: list[dict]) -> list[dict]:
    """Drop tracks with no Navidrome song ID (can't add to playlist)."""
    matched = [t for t in tracks if t.get("nav_id")]
    dropped = len(tracks) - len(matched)
    if dropped:
        log.warning("Constraints: dropped %d tracks with no Navidrome match", dropped)
    return matched


def apply_all(tracks: list[dict]) -> list[dict]:
    tracks = deduplicate(tracks)
    tracks = filter_unmatched(tracks)
    tracks = fix_consecutive_artists(tracks)
    return tracks
