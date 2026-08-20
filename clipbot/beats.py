"""Beat detection and musical-section analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from .ffmpeg import extract_audio


@dataclass
class BeatMap:
    tempo: float
    beats: list[float]          # beat times, seconds
    downbeats: list[float]      # every 4th beat (assumed 4/4)
    energy: list[float]         # per-beat RMS, normalised 0..1
    duration: float
    drop: float | None = None   # time of the biggest energy jump
    _sr: int = field(default=22050, repr=False)

    def beats_in(self, start: float, end: float) -> list[float]:
        return [b for b in self.beats if start <= b < end]

    def energy_at(self, t: float) -> float:
        """Energy of the beat nearest to t."""
        if not self.beats:
            return 0.5
        i = int(np.argmin(np.abs(np.asarray(self.beats) - t)))
        return self.energy[i]


def analyze(audio_path: Path, work_dir: Path) -> BeatMap:
    """Extract a beat grid + per-beat energy from any audio/video file."""
    wav = audio_path
    if audio_path.suffix.lower() not in {".wav", ".flac"}:
        wav = extract_audio(audio_path, work_dir / "analysis.wav")

    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    duration = float(len(y)) / sr

    # Percussive component tracks beats far more reliably on dense mixes.
    y_perc = librosa.effects.percussive(y, margin=3.0)
    tempo, frames = librosa.beat.beat_track(y=y_perc, sr=sr, trim=False)
    beats = librosa.frames_to_time(frames, sr=sr).tolist()
    tempo = float(np.atleast_1d(tempo)[0])

    if len(beats) < 4:
        # Fall back to a fixed grid so a beatless track still produces an edit.
        tempo = tempo if tempo > 30 else 120.0
        step = 60.0 / tempo
        beats = np.arange(0.0, duration, step).tolist()

    # Per-beat loudness, used to pick which effects fire where.
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    raw = np.interp(beats, rms_t, rms)
    lo, hi = float(raw.min()), float(raw.max())
    energy = ((raw - lo) / (hi - lo) if hi > lo else np.full_like(raw, 0.5)).tolist()

    downbeats = beats[::4]
    return BeatMap(
        tempo=tempo,
        beats=beats,
        downbeats=downbeats,
        energy=energy,
        duration=duration,
        drop=_find_drop(energy, beats),
    )


def _find_drop(energy: list[float], beats: list[float]) -> float | None:
    """Locate the largest sustained jump in energy — usually the drop.

    Compares the mean energy of the 8 beats before each candidate against the
    8 after, and takes the strongest positive delta in the middle of the track.
    """
    if len(energy) < 24:
        return None
    e = np.asarray(energy)
    win = 8
    best_i, best_delta = None, 0.12  # require a real jump, not noise
    for i in range(win, len(e) - win):
        delta = float(e[i:i + win].mean() - e[i - win:i].mean())
        if delta > best_delta:
            best_delta, best_i = delta, i
    return beats[best_i] if best_i is not None else None


def pick_window(bm: BeatMap, target: float) -> tuple[float, float]:
    """Choose the most energetic `target`-second window of the song.

    Snapped to a downbeat so the edit starts on a musical boundary. Prefers a
    window that contains the drop, which is what people actually clip.
    """
    if bm.duration <= target * 1.15:
        return 0.0, min(bm.duration, target)

    candidates = bm.downbeats or bm.beats
    best, best_score = (0.0, target), -1.0
    for start in candidates:
        end = start + target
        if end > bm.duration:
            break
        idx = [i for i, b in enumerate(bm.beats) if start <= b < end]
        if not idx:
            continue
        score = float(np.mean([bm.energy[i] for i in idx]))
        if bm.drop is not None and start <= bm.drop < end:
            # Bias toward windows containing the drop, and toward the drop
            # landing early enough that the payoff is on-screen.
            score += 0.35 * (1.0 - min((bm.drop - start) / target, 1.0))
        if score > best_score:
            best_score, best = score, (start, end)
    return best
