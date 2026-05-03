import logging
import json
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _librosa_features(path: str) -> Optional[dict]:
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(path, sr=None, mono=True, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        return {
            "bpm": float(tempo),
            "energy": round(min(1.0, rms * 10), 4),
            "spectral_centroid": round(spectral_centroid, 2),
            "zcr": round(zcr, 6),
            "mfcc_mean": [round(float(x), 4) for x in np.mean(mfccs, axis=1)],
        }
    except ImportError:
        log.debug("librosa not installed; skipping audio analysis")
        return None
    except Exception as exc:
        log.warning("Audio analysis failed for %s: %s", path, exc)
        return None


def analyze_tracks(tracks: list[dict], cache_forever: bool = True) -> list[dict]:
    """Enrich tracks that have a 'path' field with audio features (optional)."""
    for t in tracks:
        if t.get("audio_features"):
            continue
        path = t.get("path", "")
        if not path or not Path(path).exists():
            continue
        feats = _librosa_features(path)
        if feats:
            t["audio_features"] = feats
    return tracks
