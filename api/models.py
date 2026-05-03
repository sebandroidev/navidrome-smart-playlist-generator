from pydantic import BaseModel
from typing import Any, Optional


class HealthResponse(BaseModel):
    status: str
    last_run: dict[str, Optional[str]]
    next_run: dict[str, Optional[str]]
    library_size: int


class StatsResponse(BaseModel):
    total_tracks: int
    scored_tracks: int
    play_coverage_pct: float
    top_genres: list[dict]
    avg_score: float


class TriggerResponse(BaseModel):
    accepted: bool
    playlist_type: str
    message: str


class TriggerResult(BaseModel):
    playlist_type: str
    track_count: int
    nav_playlist_id: Optional[str]
    name: Optional[str]
    dynamic_name: Optional[str]
    duration_ms: Optional[int]


class PreviewTrack(BaseModel):
    title: Optional[str]
    artist: Optional[str]
    album: Optional[str]
    genre: Optional[str]
    score: Optional[float]
    play_count: Optional[int]
    last_played: Optional[str]


class PreviewResponse(BaseModel):
    playlist_type: str
    tracks: list[PreviewTrack]
    generated_at: str


class HistoryEntry(BaseModel):
    id: int
    playlist_type: str
    generated_at: str
    track_count: int
    nav_playlist_id: Optional[str]


class ConfigPatch(BaseModel):
    updates: dict[str, Any]
