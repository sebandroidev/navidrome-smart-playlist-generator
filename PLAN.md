# Orly Jams — Project Plan

Self-hosted smart playlist generator for Navidrome.  
Stack: Python 3.11 · FastAPI · APScheduler · SQLite · Docker

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       orly-jams container                        │
│                                                                  │
│  ┌──────────┐   ┌─────────────────────────────────────────────┐ │
│  │Scheduler │──▶│            Generation Pipeline               │ │
│  │(APSched) │   │ ingest → score → strategy → constrain → push│ │
│  └──────────┘   └─────────────────────────────────────────────┘ │
│       │                      │              │                    │
│  ┌────▼─────┐      ┌─────────▼──────┐  ┌───▼──────────────┐    │
│  │ REST API │      │   State DB     │  │    Notifier       │    │
│  │(FastAPI) │      │   (SQLite)     │  │ (Telegram webhook)│    │
│  └──────────┘      └────────────────┘  └──────────────────-┘    │
└──────────┬──────────────────────┬──────────────────────────────-┘
           │                      │
    ┌──────▼──────┐    ┌──────────▼──────────┐   ┌────────────────┐
    │  Navidrome  │    │  beets container     │   │  Ollama (opt.) │
    │  Subsonic   │    │  SQLite read-only    │   │  Gemma 2B      │
    │  API        │    │  /music read-only    │   │  (naming only) │
    └─────────────┘    └─────────────────────┘   └────────────────┘
                                 │
                       ┌─────────▼─────────┐
                       │ ListenBrainz API   │
                       │ (optional, opt-in) │
                       └───────────────────┘
```

---

## Module Structure

```
self-host smart playlist/
├── Dockerfile
├── docker-compose.yml
├── config.yml                   ← user-tunable weights + settings
├── requirements.txt
├── main.py                      ← FastAPI app + scheduler boot
│
├── ingestion/
│   ├── navidrome.py             ← Subsonic API client (paginated search3)
│   ├── beets.py                 ← SQLite reader (read-only mount)
│   ├── listenbrainz.py          ← optional LB stats + enrichment
│   └── cache.py                 ← TTL in-memory cache
│
├── scoring/
│   ├── signals.py               ← individual signal extractors
│   ├── engine.py                ← composite scorer, weight application
│   ├── audio.py                 ← optional librosa feature extraction
│   ├── similarity.py            ← cosine similarity, feature matrix
│   └── clustering.py            ← K-means genre clustering
│
├── generation/
│   ├── strategies.py            ← DailyJam + WeeklyJam generators
│   ├── constraints.py           ← dedup, no-consecutive-artist, nav_id filter
│   ├── naming.py                ← Ollama or rule-based naming
│   └── genre_mixes.py           ← genre cluster playlist generation
│
├── state/
│   └── db.py                    ← SQLite: scores, history, clusters, config
│
├── api/
│   ├── router.py                ← FastAPI routes
│   └── models.py                ← Pydantic request/response models
│
├── scheduler.py                 ← APScheduler job definitions
├── notifier.py                  ← Telegram notification sender
├── pipeline.py                  ← orchestrates ingest→score→generate→push
└── config.py                    ← config.yml loader + env var overlay
```

---

## Data Flow — Full Refresh Pipeline

```
1. INGEST
   ├── navidrome.py  → GET /rest/search3 (paginated 500/req)
   │                 → playCount, lastPlayed, starred, userRating
   └── beets.py      → SELECT items (genres, bpm, bitrate, year, path)

2. MERGE
   └── outer-join on norm(artist)::norm(title) → unified track dict

3. ENRICH (optional)
   ├── ListenBrainz  → lb_listen_count (match by artist+title)
   └── librosa       → energy, bpm, spectral_centroid, mfcc (slow, cached)

4. SCORE
   ├── play_count_signal    = log(1 + playCount) / log(1 + max_plays)
   ├── recency_signal       = exp(-days_since / halflife)
   ├── rating_signal        = userRating / 5.0
   ├── genre_affinity       = genre_cluster_weight[track.genre]
   ├── discovery_bonus      = 1.0 if never_played else 0.0
   └── lb_boost             = log(1 + lb_plays) / log(1 + max_lb)  [if enabled]

   composite_score = Σ(signal_i × weight_i)

5. PERSIST
   └── upsert tracks table (score, audio_features, lb_listen_count)

6. GENERATE
   ├── DailyStrategy (30 tracks):
   │   ├── comfort_pool  = top 60% by score, not played in 48h
   │   └── discovery     = cosine-similar to comfort seeds
   └── WeeklyStrategy (50 tracks):
       ├── comfort_pool  = top 40% by score
       └── discovery     = similar + excludes last 2 weekly playlist tracks

7. CONSTRAIN
   ├── deduplicate by track id
   ├── no two consecutive same-artist tracks (greedy swap)
   └── exclude tracks missing nav_id

8. PUSH
   └── Navidrome: find playlist by name → update song list

9. NOTIFY
   └── Telegram: "🎵 Daily Jam refreshed — 30 tracks ready"
```

---

## Scoring Weights (config.yml)

| Signal           | Default | Active when          |
|------------------|---------|----------------------|
| play_count       | 0.27    | always               |
| recency          | 0.23    | always               |
| rating           | 0.18    | always               |
| genre_affinity   | 0.14    | always               |
| discovery_bonus  | 0.08    | always               |
| lb_boost         | 0.10    | listenbrainz.enabled |

---

## REST API

| Method | Path                              | Description                              |
|--------|-----------------------------------|------------------------------------------|
| GET    | /health                           | status, last_run, next_run, library_size |
| GET    | /stats                            | track count, coverage, top genres        |
| POST   | /trigger/{daily\|weekly}          | run pipeline in background               |
| GET    | /trigger/{type}/result            | last pipeline result                     |
| GET    | /playlist/{type}/preview          | generate without pushing                 |
| GET    | /playlist/{type}/history          | recent playlist history                  |
| GET    | /config                           | current merged config                    |
| PATCH  | /config                           | hot-reload weight/cron overrides         |
| POST   | /rescan                           | ingest+score without pushing             |
| POST   | /trigger/clusters                 | run genre mix pipeline                   |
| GET    | /clusters                         | last cluster results                     |
| GET    | /clusters/preview                 | preview clusters without pushing         |

---

## State Database Schema

```sql
CREATE TABLE tracks (
    id              TEXT PRIMARY KEY,     -- norm(artist)::norm(title)
    nav_id          TEXT,
    beets_id        INTEGER,
    title           TEXT NOT NULL,
    artist          TEXT NOT NULL,
    albumartist     TEXT,
    album           TEXT,
    genre           TEXT,
    year            INTEGER,
    format          TEXT,
    bitrate         INTEGER,
    play_count      INTEGER DEFAULT 0,
    last_played     TEXT,
    starred         INTEGER DEFAULT 0,
    user_rating     INTEGER DEFAULT 0,
    composite_score REAL DEFAULT 0.0,
    audio_features  TEXT,                 -- JSON: {bpm, energy, mfcc_mean, ...}
    updated_at      TEXT NOT NULL
);

CREATE TABLE playlist_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_type   TEXT NOT NULL,        -- 'daily' | 'weekly' | 'genre_mix'
    generated_at    TEXT NOT NULL,
    track_ids       TEXT NOT NULL,        -- JSON array
    nav_playlist_id TEXT
);

CREATE TABLE genre_clusters (
    genre       TEXT PRIMARY KEY,
    cluster_id  INTEGER,
    weight      REAL DEFAULT 1.0,
    play_count  INTEGER DEFAULT 0,
    updated_at  TEXT NOT NULL
);

CREATE TABLE config_overrides (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,            -- JSON-encoded
    updated_at TEXT NOT NULL
);
```

---

## Docker Deployment (ZimaCube NAS)

```bash
docker run -d \
  --name orly-jams \
  --restart unless-stopped \
  --memory 200m --memory-swap 200m \
  -p 7070:7070 \
  --add-host host.docker.internal:host-gateway \
  -e NAVIDROME_URL=http://host.docker.internal:4533 \
  -e NAVIDROME_USER=sebastien \
  -e NAVIDROME_PASS=sebastien \
  -e BEETS_DB_PATH=/beets-db/musiclibrary.db \
  -e LB_USERNAME=<your_lb_username>      # optional
  -e TG_BOT_TOKEN=<token>                # optional
  -e TG_CHAT_ID=<chat_id>               # optional
  -v /DATA/AppData/beets/config:/beets-db:ro \
  -v /media/sdb/Musics:/music:ro \
  -v /DATA/AppData/orly-jams:/data \
  -v /DATA/homelab-setup/navidrome-smart-playlist/config.yml:/app/config.yml:ro \
  orly-jams:latest
```

Memory: 44 MB idle / 62 MB peak / 200 MB hard cap.

---

## Phase Delivery Plan

### P1 — Core ✅ DONE

| Component          | Status | Notes                                              |
|--------------------|--------|----------------------------------------------------|
| Navidrome ingest   | ✅     | Paginated search3, 500/req                         |
| beets ingest       | ✅     | Read-only SQLite, `genres` (plural) column         |
| Track merge        | ✅     | Outer-join on norm(artist)::norm(title)            |
| Scoring engine     | ✅     | 5 signals + genre affinity weights                 |
| DailyJam (30)      | ✅     | 06:00 UTC daily, 60% comfort / 40% discovery       |
| WeeklyJam (50)     | ✅     | Monday 00:00 UTC, 40% comfort / 60% discovery      |
| Navidrome push     | ✅     | Find by name → update playlist, tuple-param fix    |
| Telegram notifier  | ✅     | Opt-in via env vars                                |
| REST API           | ✅     | /health /stats /trigger /preview /config /rescan  |
| Scheduler          | ✅     | APScheduler, cron-based, daemon thread             |
| Docker             | ✅     | python:3.11-slim, 200 MB cap, OOM fix applied      |
| Rule-based naming  | ✅     | Time-slot + top-genre label                        |

---

### P2 — Intelligence ✅ DONE

| Component                    | Status | Notes                                            |
|------------------------------|--------|--------------------------------------------------|
| Cosine similarity            | ✅     | L2-normalised genre-one-hot + year/bitrate/bpm   |
| Similarity-boosted discovery | ✅     | Daily 40% / Weekly 60% via cosine centroid       |
| K-means genre clustering     | ✅     | scikit-learn, n_clusters=5, junk-genre filter    |
| Genre mix playlists          | ✅     | 5 playlists × 40 tracks, Monday 01:00 UTC        |
| Genre mix Telegram notify    | ✅     | Lists all generated playlist names               |
| /trigger/clusters endpoint   | ✅     | Route ordering fix applied                       |
| /clusters + /clusters/preview| ✅     | Returns cluster metadata + sample tracks         |

---

### P3 — AI ✅ DONE

| Component                    | Status | Notes                                            |
|------------------------------|--------|--------------------------------------------------|
| Ollama naming                | ✅     | `ollama_name()` + rule-based fallback            |
| librosa audio features       | ✅     | `scoring/audio.py` + pipeline integration        |
| audio_features → DB          | ✅     | Stored as JSON TEXT in tracks table              |
| audio_features deserialized  | ✅     | `_deserialize()` in StateDB.get_all_tracks()     |
| audio_features → similarity  | ✅     | energy, zcr, spectral_centroid dims in matrix    |
| ListenBrainz client          | ✅     | `get_top_recordings()` + `get_similar_recordings`|
| LB enrich_tracks()           | ✅     | monthly + all-time, CF name-match                |
| lb_boost scoring signal      | ✅     | `listenbrainz_signal()` in signals.py + engine   |
| lb_boost weight in config    | ✅     | config.yml + ScoringWeights.lb_boost             |
| LB pipeline integration      | ✅     | called from `ingest_and_score()` when enabled    |

---

### P4 — UX 📋 TODO

| Component                    | Status | Notes                                            |
|------------------------------|--------|--------------------------------------------------|
| Time-of-day variants         | ⬜     | Morning / Evening / Late Night playlist slots    |
| Mood playlists               | ⬜     | BPM + energy segmentation (Chill / Energy mix)   |
| Telegram inline controls     | ⬜     | /trigger /preview via bot commands               |
| Web config UI                | ⬜     | Lightweight HTML dashboard served from FastAPI   |

---

## Key Design Decisions

| Decision           | Choice                  | Rationale                                       |
|--------------------|-------------------------|-------------------------------------------------|
| Language           | Python 3.11             | librosa / scikit-learn ecosystem                |
| Web framework      | FastAPI                 | Async-native, Pydantic, auto-docs at /docs      |
| Scheduler          | APScheduler 3.x         | Embedded, no broker, cron, survives restarts    |
| State storage      | SQLite (WAL mode)       | Zero-dep, file-based, persists across restarts  |
| Beets access       | Read-only volume mount  | No lock contention with beets                   |
| Track matching     | norm(artist)::norm(title)| Handles tag discrepancies beets ↔ Navidrome    |
| Similarity         | Cosine on feature matrix| CPU-only, fast at home-library scale (<1K ms)   |
| Audio features     | librosa (opt-in)        | Heavy first-run, cached forever; off by default |
| LLM naming         | Ollama + Gemma 2B (opt-in) | Playlist naming only; works without it       |
| Memory footprint   | uvicorn + h11, no uvloop | 44 MB idle on ZimaCube NAS                    |
| CF signal          | ListenBrainz top plays  | External listen data as 6th scoring signal      |
