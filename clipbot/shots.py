"""Shot detection and ranking.

Turns a movie or scenepack into a pool of short, visually usable shots.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scenedetect import AdaptiveDetector, ContentDetector, detect

from .ffmpeg import ffmpeg_bin, probe

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts", ".wmv", ".flv"}


@dataclass
class Shot:
    src: Path
    start: float
    end: float
    score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_videos(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    return sorted(
        p for p in inp.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def split_shots(
    src: Path,
    *,
    min_len: float = 0.7,
    max_len: float = 5.0,
    threshold: float = 27.0,
    skip_intro: float = 0.0,
    skip_outro: float = 0.0,
) -> list[Shot]:
    """Cut a video into shots with PySceneDetect.

    Long shots are subdivided rather than discarded — a 40s dialogue take still
    contains usable 2s pieces, and movies are full of them.
    """
    info = probe(src)
    start_t = skip_intro if skip_intro > 0 else None
    end_t = (info.duration - skip_outro) if skip_outro > 0 else None
    if end_t is not None and start_t is not None and end_t <= start_t:
        start_t, end_t = None, None

    detector = AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=int(min_len * 15))
    try:
        scenes = detect(str(src), detector, start_time=start_t, end_time=end_t)
    except Exception:
        # AdaptiveDetector needs a decodable stream throughout; ContentDetector
        # is more forgiving of damaged or unusual files.
        scenes = detect(str(src), ContentDetector(threshold=threshold),
                        start_time=start_t, end_time=end_t)

    shots: list[Shot] = []
    for s, e in scenes:
        a, b = s.get_seconds(), e.get_seconds()
        if b - a < min_len:
            continue
        if b - a <= max_len:
            shots.append(Shot(src, a, b))
        else:
            # Subdivide, trimming the head/tail where cuts tend to be soft.
            n = int((b - a) // max_len)
            for k in range(n):
                cs = a + k * max_len
                shots.append(Shot(src, cs, min(cs + max_len, b)))
    return shots


def _sample_frames(shot: Shot, n: int = 3) -> np.ndarray | None:
    """Grab n small greyscale frames from a shot as a numpy array."""
    times = np.linspace(shot.start + 0.15, max(shot.end - 0.15, shot.start + 0.2), n)
    frames = []
    for t in times:
        proc = subprocess.run(
            [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
             "-ss", f"{t:.3f}", "-i", str(shot.src), "-frames:v", "1",
             "-vf", "scale=64:36,format=gray", "-f", "rawvideo", "-"],
            capture_output=True,
        )
        if proc.returncode == 0 and len(proc.stdout) == 64 * 36:
            frames.append(np.frombuffer(proc.stdout, dtype=np.uint8).reshape(36, 64))
    return np.stack(frames) if frames else None


def score_shots(shots: list[Shot], *, sample: bool = True) -> list[Shot]:
    """Score shots by visual interest: brightness, contrast, and motion.

    Filters out the black frames, fades, and static title cards that otherwise
    dominate a movie rip.
    """
    measured: list[tuple[Shot, float, float, float]] = []
    for sh in shots:
        if not sample:
            sh.score = 0.5
            measured.append((sh, 0.5, 0.5, 0.0))
            continue

        f = _sample_frames(sh)
        if f is None:
            continue
        fl = f.astype(np.float32)
        brightness = float(fl.mean()) / 255.0
        contrast = float(fl.std()) / 128.0
        motion = (float(np.abs(np.diff(fl, axis=0)).mean()) / 255.0
                  if len(fl) > 1 else 0.0)
        measured.append((sh, brightness, contrast, motion))

    if not measured:
        return []

    # Brightness cutoffs are relative to this source, not absolute. Graded films
    # are frequently far darker than a fixed threshold expects — Fight Club sits
    # around 0.08 mean luma, so a hardcoded 0.06 floor discards most of the
    # movie. Cut at a fraction of the source's own median instead, which drops
    # true black frames and fades while keeping deliberately dark photography.
    brights = np.array([m[1] for m in measured])
    median = float(np.median(brights))
    dark_floor = max(0.012, median * 0.35)
    # Reference point for the "well exposed" term, likewise source-relative.
    target = float(np.clip(median, 0.10, 0.55))

    kept: list[Shot] = []
    for sh, brightness, contrast, motion in measured:
        if brightness < dark_floor or contrast < 0.045:
            continue  # black frame, fade, or flat card

        sh.score = (
            0.34 * min(contrast * 2.2, 1.0)
            + 0.46 * min(motion * 12.0, 1.0)
            + 0.20 * (1.0 - min(abs(brightness - target) / max(target, 0.1), 1.0))
        )
        kept.append(sh)

    kept.sort(key=lambda s: s.score, reverse=True)
    return kept


def diversify(shots: list[Shot], count: int, *, min_gap: float = 6.0) -> list[Shot]:
    """Pick `count` high-scoring shots that aren't all from the same moment.

    Without this a movie's single most action-heavy minute supplies every shot
    and the edit looks like a loop.
    """
    picked: list[Shot] = []
    for sh in shots:  # already sorted by score
        if len(picked) >= count:
            break
        if any(p.src == sh.src and abs(p.start - sh.start) < min_gap for p in picked):
            continue
        picked.append(sh)

    # Relax the spacing rule if the source was too short to fill the quota.
    if len(picked) < count:
        for sh in shots:
            if len(picked) >= count:
                break
            if sh not in picked:
                picked.append(sh)
    return picked[:count]
