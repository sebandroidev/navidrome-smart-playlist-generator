import logging
import urllib.parse
import requests
from typing import Optional

log = logging.getLogger(__name__)

_PAGE_SIZE = 500


class NavidromeClient:
    def __init__(self, url: str, user: str, password: str):
        self._base = url.rstrip("/")
        self._auth = {
            "u": user, "p": password,
            "v": "1.16.1", "c": "orly-jams", "f": "json",
        }

    def _get(self, endpoint: str, **params) -> dict:
        url = f"{self._base}/rest/{endpoint}"
        resp = requests.get(url, params={**self._auth, **params}, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        sr = body.get("subsonic-response", {})
        if sr.get("status") != "ok":
            raise RuntimeError(f"Subsonic error on {endpoint}: {sr.get('error', sr)}")
        return sr

    # ── library fetch ─────────────────────────────────────────────────────────

    def get_all_songs(self) -> list[dict]:
        """Paginate through search3 to pull every song with its stats."""
        songs: list[dict] = []
        offset = 0
        while True:
            try:
                sr = self._get(
                    "search3",
                    query="",
                    songCount=_PAGE_SIZE,
                    songOffset=offset,
                    albumCount=0,
                    artistCount=0,
                )
            except Exception as exc:
                log.error("Navidrome search3 failed at offset %d: %s", offset, exc)
                break
            batch = sr.get("searchResult3", {}).get("song", [])
            songs.extend(batch)
            if len(batch) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        log.info("Navidrome: fetched %d songs", len(songs))
        return songs

    # ── playlist management ───────────────────────────────────────────────────

    def get_playlists(self) -> list[dict]:
        sr = self._get("getPlaylists")
        return sr.get("playlists", {}).get("playlist", [])

    def find_playlist_id(self, name: str) -> Optional[str]:
        for pl in self.get_playlists():
            if pl.get("name") == name:
                return pl.get("id")
        return None

    def push_playlist(self, name: str, song_ids: list[str]) -> str:
        """Create or replace a named playlist. Returns the playlist ID."""
        existing_id = self.find_playlist_id(name)

        if existing_id:
            # Clear existing songs by removing all indices
            pl_detail = self._get("getPlaylist", id=existing_id)
            existing_entries = pl_detail.get("playlist", {}).get("entry", [])
            if existing_entries:
                params = list(self._auth_pairs()) + [("playlistId", existing_id)]
                for i in range(len(existing_entries)):
                    params.append(("songIndexToRemove", str(i)))
                resp = requests.get(
                    f"{self._base}/rest/updatePlaylist", params=params, timeout=30
                )
                resp.raise_for_status()

            # Add new songs
            self._add_songs_to_playlist(existing_id, song_ids)
            log.info("Navidrome: replaced playlist '%s' (%s) → %d tracks",
                     name, existing_id, len(song_ids))
            return existing_id
        else:
            # Create new playlist with first batch of songs
            params = list(self._auth_pairs()) + [("name", name)]
            for sid in song_ids[:200]:
                params.append(("songId", sid))
            resp = requests.get(
                f"{self._base}/rest/createPlaylist", params=params, timeout=30
            )
            resp.raise_for_status()
            sr = resp.json().get("subsonic-response", {})
            pl_id = str(sr.get("playlist", {}).get("id", ""))
            if pl_id and len(song_ids) > 200:
                self._add_songs_to_playlist(pl_id, song_ids[200:])
            log.info("Navidrome: created playlist '%s' (%s) → %d tracks",
                     name, pl_id, len(song_ids))
            return pl_id

    def _auth_pairs(self) -> list[tuple]:
        return [(k, v) for k, v in self._auth.items()]

    def _add_songs_to_playlist(self, playlist_id: str, song_ids: list[str]):
        for chunk_start in range(0, len(song_ids), 100):
            chunk = song_ids[chunk_start:chunk_start + 100]
            params = [("u", self._auth["u"]), ("p", self._auth["p"]),
                      ("v", self._auth["v"]), ("c", self._auth["c"]),
                      ("f", self._auth["f"]), ("playlistId", playlist_id)]
            for sid in chunk:
                params.append(("songId", sid))
            resp = requests.get(
                f"{self._base}/rest/updatePlaylist", params=params, timeout=30
            )
            resp.raise_for_status()
