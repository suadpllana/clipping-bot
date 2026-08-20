"""Highlight mode: pick the best N moments of a source and cut them straight.

This is the "upload an episode, get clips" path. Unlike the music path it keeps
the source's own audio, so a clip is a real excerpt — dialogue intact, in sync —
rather than a montage. Selection is driven by the coarse `Timeline` from
`analyze`, so the whole film is only decoded once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analyze import Timeline, refine_cuts
from .ffmpeg import probe, run

# Output geometry per aspect. Vertical is the default because that is what the
# format is for; the others exist because not every clip is going to TikTok.
ASPECTS = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}


@dataclass
class Clip:
    index: int
    start: float
    end: float
    score: float
    # (time_from_clip_start, horizontal_crop_position 0..1) — steps at shot cuts
    framing: list[tuple[float, float]] = field(default_factory=list)
    path: Path | None = None
    thumb: Path | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def _interest(tl: Timeline) -> np.ndarray:
    """Per-sample "is something happening here" score, 0..1.

    Visual movement alone ranks the credits scroll and a shaky establishing
    shot as highly as a fight. Audio is what disambiguates: loudness says
    something is playing, and local dynamics say something is *happening* —
    dialogue and impacts modulate, room tone and a score drone do not.
    """
    vis = 0.62 * tl.motion + 0.38 * tl.contrast

    if tl.has_audio and float(tl.loudness.max()) > 0.02:
        aud = 0.45 * tl.loudness + 0.55 * tl.dynamics
        score = 0.48 * vis + 0.52 * aud
    else:
        score = vis

    # Cut rate as a proxy for editorial intensity — a densely cut passage is
    # almost always a set piece or an emotional peak.
    if len(tl.cuts) > 2:
        n = len(tl.brightness)
        rate = np.zeros(n, dtype=np.float32)
        idx = np.clip((tl.cuts * tl.fps).astype(int), 0, n - 1)
        rate[idx] = 1.0
        k = max(int(tl.fps * 6), 3)
        kern = np.ones(k, dtype=np.float32) / k
        dens = np.convolve(rate, kern, mode="same")
        hi = float(dens.max())
        if hi > 1e-6:
            score = score + 0.14 * (dens / hi)

    # Black frames, fades and flat title cards are never the clip.
    dark = tl.brightness < max(0.02, float(np.median(tl.brightness)) * 0.3)
    score = np.where(dark | (tl.contrast < 0.03), score * 0.05, score)

    # Smooth so a single loud frame cannot win a window on its own.
    k = max(int(tl.fps * 1.5), 3)
    kern = np.ones(k, dtype=np.float32) / k
    score = np.convolve(score, kern, mode="same")

    hi = float(score.max())
    return score / hi if hi > 1e-6 else score


def select(
    tl: Timeline,
    *,
    count: int,
    length: float,
    skip_intro: float = 0.0,
    skip_outro: float = 0.0,
    min_gap: float | None = None,
) -> list[Clip]:
    """Choose `count` non-overlapping windows of `length` seconds.

    Greedy on window score with an enforced gap, which spreads picks across the
    runtime. Taking the literal top N instead would hand back five clips of the
    same two-minute battle.
    """
    interest = _interest(tl)
    fps = tl.fps
    win = max(int(round(length * fps)), 2)

    if win >= len(interest):
        return [Clip(index=0, start=0.0, end=min(length, tl.duration), score=1.0)]

    # Rolling mean of the interest signal — the score of a window starting at i.
    cum = np.concatenate([[0.0], np.cumsum(interest)])
    scores = (cum[win:] - cum[:-win]) / win

    lo = int(skip_intro * fps)
    hi = len(scores) - int(skip_outro * fps)
    if hi - lo < 1:
        lo, hi = 0, len(scores)
    mask = np.zeros(len(scores), dtype=bool)
    mask[max(lo, 0):min(hi, len(scores))] = True

    usable = (min(hi, len(scores)) - max(lo, 0)) / fps
    if min_gap is None:
        # Spread across the runtime without being so strict that a short source
        # cannot fill the quota.
        min_gap = max(length * 1.15, min(usable / max(count, 1) * 0.62, 600.0))
    gap = int(min_gap * fps)

    order = np.argsort(scores)[::-1]
    picks: list[int] = []
    for i in order:
        if len(picks) >= count:
            break
        if not mask[i]:
            continue
        if any(abs(int(i) - p) < gap for p in picks):
            continue
        picks.append(int(i))

    # Source too short (or too uniform) to honour the spacing — relax it rather
    # than return fewer clips than asked for.
    if len(picks) < count:
        relaxed = int(length * fps * 1.02)
        for i in order:
            if len(picks) >= count:
                break
            i = int(i)
            if not mask[i] or i in picks:
                continue
            if any(abs(i - p) < relaxed for p in picks):
                continue
            picks.append(i)

    picks.sort()
    return [
        Clip(index=n, start=p / fps, end=p / fps + length, score=float(scores[p]))
        for n, p in enumerate(picks)
    ]


def snap_and_frame(src: Path, tl: Timeline, clips: list[Clip],
                   *, snap: float = 1.2) -> list[Clip]:
    """Align each clip to a real shot boundary and work out its framing.

    A clip that opens three frames into a shot reads as a mistake, so the start
    is pulled to the nearest cut when one is close enough. Framing is then
    resolved per shot inside the clip, because the subject moves between shots
    and a single crop position for the whole clip loses them.
    """
    total = tl.duration
    for c in clips:
        cuts = refine_cuts(src, c.start, c.end)
        if cuts:
            near = min(cuts, key=lambda t: abs(t - c.start))
            if abs(near - c.start) <= snap:
                # `duration` is derived from start/end, so it has to be read
                # before start moves — otherwise the clip silently grows or
                # shrinks by however far it was snapped.
                want = c.duration
                c.start = max(0.0, min(near, max(total - want, 0.0)))
                c.end = c.start + want
                shift = near - c.start
                cuts = [t - shift for t in cuts]

        inner = [t for t in cuts if c.start + 0.25 < t < c.end - 0.25]
        c.framing = _framing(tl, c, inner)
    return clips


def _framing(tl: Timeline, clip: Clip, cuts: list[float]) -> list[tuple[float, float]]:
    """Crop position per shot: (offset from clip start, position 0..1).

    Held constant within a shot and stepped at cuts — the step is invisible
    because it happens on a cut, whereas a continuously moving crop reads as a
    drifting, unmotivated camera move.
    """
    bounds = [clip.start] + sorted(cuts) + [clip.end]
    out: list[tuple[float, float]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < 0.2:
            continue
        i, j = tl.index(a), tl.index(b)
        seg = tl.centroid[i:max(j, i + 1)]
        if not len(seg):
            continue
        # Median, not mean: one frame of a bright object crossing the far edge
        # should not pull the whole shot's framing with it.
        pos = float(np.median(seg))
        # Stay away from the extremes; a hard-edge crop looks like a mistake
        # even when the subject really is at the edge of frame.
        out.append((round(a - clip.start, 3), float(np.clip(pos, 0.14, 0.86))))

    return out or [(0.0, 0.5)]


def _crop_x(framing: list[tuple[float, float]]) -> str:
    """Build a crop-x expression from the framing steps.

    ffmpeg expressions have no arrays, so a piecewise constant is a nest of
    if()s. Capped at a sane number of branches — the expression is parsed per
    frame and long ones cost real time for no visible gain.
    """
    steps = framing[:14]
    if len(steps) == 1:
        return f"(iw-ow)*{steps[0][1]:.4f}"

    # Each entry is the START of a shot, so a shot's position has to be guarded
    # by where the *next* one begins — testing against its own start time would
    # hold each position for no time at all and shift the whole clip's framing
    # one shot early. `t` runs from the clip start because setpts rebases it.
    expr = f"{steps[-1][1]:.4f}"
    for i in range(len(steps) - 2, -1, -1):
        expr = f"if(lt(t,{steps[i + 1][0]:.3f}),{steps[i][1]:.4f},{expr})"
    return f"(iw-ow)*({expr})"


def render_clip(
    src: Path, clip: Clip, dst: Path, *,
    aspect: str = "9:16",
    crf: int = 19,
    sharpen: bool = True,
    has_audio: bool = True,
    normalize_audio: bool = True,
    fade: float = 0.12,
) -> Path:
    """Cut and encode one clip, audio in sync, ready to upload."""
    w, h = ASPECTS.get(aspect, ASPECTS["9:16"])

    vf = [
        # Rebase timestamps to zero first. A fast seek lands the video on the
        # next whole frame but the audio on the next packet, which leaves the
        # two streams starting tens of milliseconds apart in the container —
        # enough to read as a lip-sync error. It also makes `t` in the filters
        # below measure from the clip start, which the framing expression and
        # the fades both rely on.
        "setpts=PTS-STARTPTS",
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={w}:{h}:x='{_crop_x(clip.framing)}':y=(ih-oh)/2",
        "setsar=1",
    ]
    if sharpen:
        # Cropping 16:9 into 9:16 means upscaling ~1.8x; a light luma sharpen
        # restores the edge definition that costs, without ringing.
        vf.append("unsharp=5:5:0.5:5:5:0.0")
    vf.append("format=yuv420p")
    # Tag the colour space on the frames themselves. Passing -colorspace and
    # friends as encoder options only lands the matrix in this ffmpeg build;
    # untagged primaries/transfer are why a re-uploaded clip can come back
    # looking washed out.
    vf.append("setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709")
    if fade > 0:
        vf.append(f"fade=t=in:st=0:d={fade:.2f}")
        vf.append(f"fade=t=out:st={max(clip.duration - fade, 0):.3f}:d={fade:.2f}")

    args = [
        "-accurate_seek", "-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}",
        "-i", str(src),
    ]
    maps = ["-map", "0:v:0"]

    if has_audio:
        af = ["asetpts=PTS-STARTPTS",
              f"afade=t=in:st=0:d={min(fade, 0.08):.2f}",
              f"afade=t=out:st={max(clip.duration - 0.08, 0):.3f}:d=0.08"]
        if normalize_audio:
            # Social platforms normalise on ingest anyway; getting there first
            # means they do not have to pull the whole clip down.
            af.append("loudnorm=I=-14:TP=-1.5:LRA=11")
        args += ["-af", ",".join(af)]
        maps += ["-map", "0:a:0?"]
    else:
        # A silent track beats no track: some players and uploaders choke on a
        # video-only file.
        args += ["-f", "lavfi", "-t", f"{clip.duration:.3f}",
                 "-i", "anullsrc=r=48000:cl=stereo"]
        maps = ["-map", "0:v:0", "-map", "1:a:0"]

    args += [
        "-vf", ",".join(vf), *maps,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
        "-x264-params", "keyint=120:min-keyint=48:scenecut=40",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        str(dst),
    ]
    run(args)
    clip.path = dst
    return dst


def make_thumb(clip_path: Path, dst: Path, *, width: int = 320) -> Path | None:
    """Poster frame from the rendered clip, for the results grid."""
    try:
        run(["-ss", "0.6", "-i", str(clip_path), "-frames:v", "1",
             "-vf", f"scale={width}:-2:flags=lanczos", "-q:v", "4", str(dst)])
        return dst
    except RuntimeError:
        return None


def source_has_audio(src: Path) -> bool:
    try:
        return probe(src).has_audio
    except Exception:
        return False
