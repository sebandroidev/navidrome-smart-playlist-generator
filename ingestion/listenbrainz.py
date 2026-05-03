import logging
import requests
from typing import Optional

log = logging.getLogger(__name__)

_BASE = "https://api.listenbrainz.org/1"


class ListenBrainzClient:
    def __init__(self, username: str):
        self._username = username
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "orly-jams/1.0"

    def get_top_recordings(self, time_range: str = "month", count: int = 100) -> list[dict]:
        """Returns list of {recording_mbid, track_name, artist_name, listen_count}."""
        try:
            r = self._session.get(
                f"{_BASE}/stats/user/{self._username}/recordings",
                params={"range": time_range, "count": count},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            recordings = (
                data.get("payload", {})
                    .get("recordings", [])
            )
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

    def get_cf_recommendations(self, count: int = 100) -> list[dict]:
        """Collaborative-filter recommendations from ListenBrainz."""
        try:
            r = self._session.get(
                f"{_BASE}/cf/recommendation/user/{self._username}/recording",
                params={"count": count},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            recs = (
                data.get("payload", {})
                    .get("mbid_mapping", {})
            )
            # payload varies by API version; handle both formats
            if isinstance(recs, list):
                return recs
            return []
        except Exception as exc:
            log.warning("ListenBrainz CF recommendations failed: %s", exc)
            return []
