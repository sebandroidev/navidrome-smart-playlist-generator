import re
import logging
import unicodedata
import requests

log = logging.getLogger(__name__)

_BASE = "https://api.listenbrainz.org/1"


def _norm_key(artist: str, title: str) -> str:
    """Same normalization as pipeline merge key — used to match LB tracks to local."""
    def _n(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
        return re.sub(r"[^a-z0-9]", "", s)
    a, t = _n(artist), _n(title)
    return f"{a}::{t}" if a and t else ""


class ListenBrainzClient:
    def __init__(self, username: str):
        self._username = username
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "orly-jams/1.0"

    def get_top_recordings(self, time_range: str = "month", count: int = 200) -> list[dict]:
        """Returns [{recording_mbid, track_name, artist_name, listen_count}]."""
        try:
            r = self._session.get(
                f"{_BASE}/stats/user/{self._username}/recordings",
                params={"range": time_range, "count": count},
                timeout=15,
            )
            r.raise_for_status()
            recordings = r.json().get("payload", {}).get("recordings", [])
            return [
                {
                    "recording_mbid": rec.get("recording_mbid", ""),
                    "track_name":     rec.get("track_name", ""),
                    "artist_name":    rec.get("artist_name", ""),
                    "listen_count":   rec.get("listen_count", 0),
                }
                for rec in recordings
            ]
        except Exception as exc:
            log.warning("ListenBrainz top_recordings failed: %s", exc)
            return []

    def get_similar_recordings(self, count: int = 200) -> list[dict]:
        """
        Fetch troi-bot / cf recommendation recordings.
        Returns [{track_name, artist_name}] for name-matching against local library.
        """
        try:
            r = self._session.get(
                f"{_BASE}/cf/recommendation/user/{self._username}/recording",
                params={"count": count},
                timeout=15,
            )
            r.raise_for_status()
            payload = r.json().get("payload", {})
            recs = payload.get("recording_list", payload.get("mbid_mapping", []))
            if not isinstance(recs, list):
                return []
            # Each item may have artist_name / track_name or just recording_mbid
            results = []
            for rec in recs:
                if isinstance(rec, dict):
                    results.append({
                        "track_name":  rec.get("track_name") or rec.get("recording_name", ""),
                        "artist_name": rec.get("artist_name") or rec.get("artist_credit_name", ""),
                    })
            return results
        except Exception as exc:
            log.warning("ListenBrainz CF recommendations failed: %s", exc)
            return []


def enrich_tracks(tracks: list[dict], cfg) -> list[dict]:
    """
    Fetch LB listen stats and CF recs; inject lb_listen_count and lb_cf_rec into tracks.
    Call this before scoring so the lb_boost signal has data to work with.
    No-ops silently if LB is unreachable.
    """
    if not cfg.listenbrainz.username:
        return tracks

    client = ListenBrainzClient(cfg.listenbrainz.username)

    # Fetch monthly + all-time top recordings; take the max listen_count per track
    monthly  = client.get_top_recordings("month",    200)
    alltime  = client.get_top_recordings("all_time", 500)

    lb_plays: dict[str, int] = {}
    for rec in alltime + monthly:
        key = _norm_key(rec["artist_name"], rec["track_name"])
        if key:
            lb_plays[key] = max(lb_plays.get(key, 0), rec["listen_count"])

    # Fetch CF recommendations and build a set of norm keys we can match locally
    cf_recs   = client.get_similar_recordings(200)
    cf_keys: set[str] = set()
    for rec in cf_recs:
        key = _norm_key(rec.get("artist_name", ""), rec.get("track_name", ""))
        if key:
            cf_keys.add(key)

    lb_matched = 0
    cf_matched  = 0
    for t in tracks:
        key = _norm_key(t.get("artist", ""), t.get("title", ""))
        lb_count = lb_plays.get(key, 0)
        t["lb_listen_count"] = lb_count
        t["lb_cf_rec"]       = key in cf_keys
        if lb_count > 0:
            lb_matched += 1
        if t["lb_cf_rec"]:
            cf_matched += 1

    log.info(
        "ListenBrainz: %d/%d tracks with listen data, %d CF recs matched",
        lb_matched, len(tracks), cf_matched,
    )
    return tracks
