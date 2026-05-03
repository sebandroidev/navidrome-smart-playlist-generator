import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/app/config.yml"))
# fallback for local dev: config.yml next to this file
if not _CONFIG_PATH.exists():
    _CONFIG_PATH = Path(__file__).parent / "config.yml"


@dataclass
class NavidromeConfig:
    url: str = "http://host.docker.internal:4533"
    user: str = "sebastien"
    password: str = "sebastien"


@dataclass
class BeetsConfig:
    db_path: str = "/beets-db/musiclibrary.db"


@dataclass
class ListenBrainzConfig:
    enabled: bool = False
    username: str = ""


@dataclass
class OllamaConfig:
    enabled: bool = False
    host: str = "http://host.docker.internal:11434"
    model: str = "gemma2:2b"


@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


@dataclass
class ScoringWeights:
    play_count: float = 0.30
    recency: float = 0.25
    rating: float = 0.20
    genre_affinity: float = 0.15
    discovery_bonus: float = 0.10


@dataclass
class ScoringConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    recency_halflife_days: float = 7.0


@dataclass
class DailyGenConfig:
    cron: str = "0 6 * * *"
    track_count: int = 30
    comfort_ratio: float = 0.60
    exclude_played_within_hours: int = 48


@dataclass
class WeeklyGenConfig:
    cron: str = "0 0 * * 1"
    track_count: int = 50
    comfort_ratio: float = 0.40
    exclude_last_n_weekly_playlists: int = 2


@dataclass
class AudioAnalysisConfig:
    enabled: bool = False
    cache_forever: bool = True


@dataclass
class AppConfig:
    navidrome: NavidromeConfig = field(default_factory=NavidromeConfig)
    beets: BeetsConfig = field(default_factory=BeetsConfig)
    listenbrainz: ListenBrainzConfig = field(default_factory=ListenBrainzConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    daily: DailyGenConfig = field(default_factory=DailyGenConfig)
    weekly: WeeklyGenConfig = field(default_factory=WeeklyGenConfig)
    audio_analysis: AudioAnalysisConfig = field(default_factory=AudioAnalysisConfig)
    state_db_path: str = "/data/orly_jams.db"
    playlist_names: dict = field(default_factory=lambda: {
        "daily": "🎵 Daily Jam",
        "weekly": "🗓 Weekly Jam",
    })


def _load_raw() -> dict:
    raw: dict = {}
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            raw = yaml.safe_load(f) or {}

    # env var overrides (only applied if set)
    env_nav_url = os.environ.get("NAVIDROME_URL")
    env_nav_user = os.environ.get("NAVIDROME_USER")
    env_nav_pass = os.environ.get("NAVIDROME_PASS")
    env_tg_token = os.environ.get("TG_BOT_TOKEN")
    env_tg_chat = os.environ.get("TG_CHAT_ID")
    env_ollama = os.environ.get("OLLAMA_HOST")
    env_beets = os.environ.get("BEETS_DB_PATH")
    env_lb = os.environ.get("LB_USERNAME")
    env_state = os.environ.get("STATE_DB_PATH")

    nav = raw.setdefault("navidrome", {})
    if env_nav_url:  nav["url"] = env_nav_url
    if env_nav_user: nav["user"] = env_nav_user
    if env_nav_pass: nav["pass"] = env_nav_pass

    tg = raw.setdefault("telegram", {})
    if env_tg_token:
        tg["bot_token"] = env_tg_token
        tg["enabled"] = True
    if env_tg_chat:
        tg["chat_id"] = env_tg_chat

    if env_ollama:
        ol = raw.setdefault("ollama", {})
        ol["host"] = env_ollama
        ol["enabled"] = True

    if env_beets:
        raw.setdefault("beets", {})["db_path"] = env_beets

    if env_lb:
        lb = raw.setdefault("listenbrainz", {})
        lb["username"] = env_lb
        lb["enabled"] = True

    if env_state:
        raw["state_db_path"] = env_state

    return raw


def _parse(raw: dict) -> AppConfig:
    cfg = AppConfig()

    n = raw.get("navidrome", {})
    cfg.navidrome = NavidromeConfig(
        url=n.get("url", cfg.navidrome.url),
        user=n.get("user", cfg.navidrome.user),
        password=n.get("pass", n.get("password", cfg.navidrome.password)),
    )

    b = raw.get("beets", {})
    cfg.beets = BeetsConfig(db_path=b.get("db_path", cfg.beets.db_path))

    lb = raw.get("listenbrainz", {})
    cfg.listenbrainz = ListenBrainzConfig(
        enabled=lb.get("enabled", False),
        username=lb.get("username", ""),
    )

    o = raw.get("ollama", {})
    cfg.ollama = OllamaConfig(
        enabled=o.get("enabled", False),
        host=o.get("host", cfg.ollama.host),
        model=o.get("model", cfg.ollama.model),
    )

    t = raw.get("telegram", {})
    cfg.telegram = TelegramConfig(
        enabled=t.get("enabled", False),
        bot_token=t.get("bot_token", ""),
        chat_id=str(t.get("chat_id", "")),
    )

    s = raw.get("scoring", {})
    w = s.get("weights", {})
    cfg.scoring = ScoringConfig(
        weights=ScoringWeights(
            play_count=float(w.get("play_count", 0.30)),
            recency=float(w.get("recency", 0.25)),
            rating=float(w.get("rating", 0.20)),
            genre_affinity=float(w.get("genre_affinity", 0.15)),
            discovery_bonus=float(w.get("discovery_bonus", 0.10)),
        ),
        recency_halflife_days=float(s.get("recency_halflife_days", 7.0)),
    )

    sch = raw.get("scheduling", {})
    dj = sch.get("daily_jam", {})
    wj = sch.get("weekly_jam", {})
    gen = raw.get("generation", {})
    dg = gen.get("daily", {})
    wg = gen.get("weekly", {})

    cfg.daily = DailyGenConfig(
        cron=dj.get("cron", cfg.daily.cron),
        track_count=int(dj.get("track_count", cfg.daily.track_count)),
        comfort_ratio=float(dg.get("comfort_ratio", cfg.daily.comfort_ratio)),
        exclude_played_within_hours=int(
            dg.get("exclude_played_within_hours", cfg.daily.exclude_played_within_hours)
        ),
    )

    cfg.weekly = WeeklyGenConfig(
        cron=wj.get("cron", cfg.weekly.cron),
        track_count=int(wj.get("track_count", cfg.weekly.track_count)),
        comfort_ratio=float(wg.get("comfort_ratio", cfg.weekly.comfort_ratio)),
        exclude_last_n_weekly_playlists=int(
            wg.get("exclude_last_n_weekly_playlists", cfg.weekly.exclude_last_n_weekly_playlists)
        ),
    )

    aa = raw.get("audio_analysis", {})
    cfg.audio_analysis = AudioAnalysisConfig(
        enabled=aa.get("enabled", False),
        cache_forever=aa.get("cache_forever", True),
    )

    if "state_db_path" in raw:
        cfg.state_db_path = raw["state_db_path"]

    if "playlist_names" in raw:
        cfg.playlist_names.update(raw["playlist_names"])

    return cfg


_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _instance
    if _instance is None:
        _instance = _parse(_load_raw())
    return _instance


def reload_config() -> AppConfig:
    global _instance
    _instance = _parse(_load_raw())
    return _instance


def patch_config(updates: dict) -> AppConfig:
    """Apply dot-notation key updates (e.g. 'scoring.weights.play_count': 0.4)."""
    raw = _load_raw()
    for dotkey, val in updates.items():
        parts = dotkey.split(".")
        d = raw
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    global _instance
    _instance = _parse(raw)
    return _instance
