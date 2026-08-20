"""Cut planning: decide what appears on screen, and for how long.

Pure data. Nothing is rendered here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from .beats import BeatMap
from .shots import Shot


@dataclass
class Segment:
    src: Path
    src_start: float
    duration: float        # output duration, seconds
    effects: list[str] = field(default_factory=list)
    speed: float = 1.0     # >1 renders more source than duration (fast-forward)
    energy: float = 0.5


# Beat multiples allowed per section, by intensity. 1 = cut on every beat.
_SLOW, _MED, _FAST = (4, 2), (2, 2, 1), (1, 1, 2)


def _stride_for(energy: float, rng: random.Random) -> int:
    if energy < 0.35:
        return rng.choice(_SLOW)
    if energy < 0.68:
        return rng.choice(_MED)
    return rng.choice(_FAST)


def build_plan(
    bm: BeatMap,
    pool: list[Shot],
    *,
    window: tuple[float, float],
    seed: int = 0,
    max_stride: int | None = None,
) -> list[Segment]:
    """Map the beat grid onto shots, one segment per cut.

    Cut density follows the music: sparse beats in quiet sections, every beat
    once the energy comes up. Effects are assigned from the same energy signal
    so hits land with the track rather than on a timer.
    """
    rng = random.Random(seed)
    start, end = window
    grid = [b for b in bm.beats if start <= b <= end]
    if len(grid) < 2:
        step = 60.0 / max(bm.tempo, 60.0)
        grid = [start + i * step for i in range(int((end - start) / step) + 1)]

    if not pool:
        raise ValueError("No usable shots were found in the source video.")

    segments: list[Segment] = []
    used: set[tuple[Path, float]] = set()
    i = 0
    while i < len(grid) - 1:
        t = grid[i]
        energy = bm.energy_at(t)
        stride = _stride_for(energy, rng)
        if max_stride:
            stride = min(stride, max_stride)
        j = min(i + stride, len(grid) - 1)
        dur = grid[j] - t
        if dur < 0.12:                      # guard against duplicate beat times
            i = j if j > i else i + 1
            continue
        # Very fast tempos can produce sub-frame cuts; keep them watchable.
        dur = max(dur, 0.18)

        shot = _next_shot(pool, used, rng, dur)
        seg = Segment(
            src=shot.src,
            src_start=_pick_in_shot(shot, dur, rng),
            duration=dur,
            energy=energy,
        )
        seg.effects = _effects_for(energy, len(segments), bm, t, rng)
        segments.append(seg)
        i = j

    return segments


def _next_shot(
    pool: list[Shot], used: set[tuple[Path, float]],
    rng: random.Random, dur: float,
) -> Shot:
    """Take the best-scoring unused shot that's long enough; recycle if needed."""
    for sh in pool:
        key = (sh.src, sh.start)
        if key in used:
            continue
        if sh.duration + 0.05 >= dur:
            used.add(key)
            return sh
    # Pool exhausted — reuse, preferring shots that can cover the duration.
    fits = [s for s in pool if s.duration + 0.05 >= dur] or pool
    return rng.choice(fits)


def _pick_in_shot(shot: Shot, dur: float, rng: random.Random) -> float:
    """Choose where inside a shot to start, avoiding the soft edges of a cut."""
    slack = shot.duration - dur
    if slack <= 0.1:
        return shot.start
    lead = min(0.12, slack / 2)
    return shot.start + lead + rng.random() * max(slack - lead * 2, 0.0)


def _effects_for(
    energy: float, index: int, bm: BeatMap, t: float, rng: random.Random,
) -> list[str]:
    """Assign per-segment effects. Restraint matters — everything on every cut
    reads as noise, so effects scale with energy and stay mutually exclusive
    where they'd fight each other.
    """
    fx: list[str] = []

    # The drop gets an unmistakable marker.
    if bm.drop is not None and abs(t - bm.drop) < 0.25:
        return ["flash", "punch"]

    # Deliberate breathing room. Even in a loud section a fraction of cuts stay
    # completely clean — back-to-back effects on every single cut stop reading
    # as emphasis and start reading as noise, and the contrast is what makes
    # the hits land.
    if rng.random() < (0.18 if energy > 0.72 else 0.30):
        return []

    if energy > 0.72:
        fx.append(rng.choices(["punch", "shake", "zoomout"], [0.55, 0.25, 0.20])[0])
    elif energy > 0.45:
        if rng.random() < 0.55:
            fx.append(rng.choice(["punch", "driftin", "driftout"]))
    else:
        if rng.random() < 0.45:
            fx.append(rng.choice(["driftin", "driftout"]))

    # Stacked accents only on top of an existing move, and never both at once.
    if fx and energy > 0.8 and rng.random() < 0.20:
        fx.append("rgbsplit")
    elif fx and index > 0 and energy > 0.6 and rng.random() < 0.15:
        fx.append("flash")
    return fx
