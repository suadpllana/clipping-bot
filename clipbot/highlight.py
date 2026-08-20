"""Highlight mode: pick the best N moments of a source and cut them straight.

This is the "upload an episode, get clips" path. Unlike the music path it keeps
the source's own audio, so a clip is a real excerpt — dialogue intact, in sync —
rather than a montage. Selection is driven by the coarse `Timeline` from
`analyze` and the structure map from `segments`, so the whole film is only
decoded once.

Two kinds of moment are worth clipping and they look nothing alike. A fight is
motion, impacts and fast cutting. A scene worth listening to is the opposite:
somebody talking, held shots, the energy in the delivery rather than the
frame. Scoring them on one blended curve finds neither — it rewards whatever
is merely busy, which is why the credits and the opening theme used to win. So
each is scored on its own terms and a window is judged by whichever it is
better at.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analyze import Timeline, refine_cuts
from .ffmpeg import probe, run
from .segments import Structure

# Output geometry per aspect. Vertical is the default because that is what the
# format is for; the others exist because not every clip is going to TikTok.
ASPECTS = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

# How a source that is the wrong shape for the output gets there.
#   fit   whole frame, scaled to fit, blurred copy of itself behind it
#   fill  scaled up and cropped to the action — loses the sides
#   pad   whole frame on flat black
FRAMES = ("auto", "fit", "fill", "pad")

# Below this much of the frame surviving a fill-crop, `auto` stops cropping.
# 16:9 into 9:16 keeps 32% — two thirds of every shot thrown away, which is
# what makes a cropped clip read as a mistake rather than a framing choice.
_FILL_FLOOR = 0.88


@dataclass
class Clip:
    index: int
    start: float
    end: float
    score: float
    kind: str = "moment"          # fight | talk | moment
    # (time_from_clip_start, horizontal_crop_position 0..1) — steps at shot cuts
    framing: list[tuple[float, float]] = field(default_factory=list)
    path: Path | None = None
    thumb: Path | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------
# what makes a moment
# --------------------------------------------------------------------------

def _cut_rate(tl: Timeline) -> np.ndarray:
    """Cuts per second, smoothed — a proxy for editorial intensity."""
    n = len(tl.brightness)
    if len(tl.cuts) < 3:
        return np.zeros(n, dtype=np.float32)
    rate = np.zeros(n, dtype=np.float32)
    idx = np.clip((tl.cuts * tl.fps).astype(int), 0, n - 1)
    rate[idx] = 1.0
    k = max(int(tl.fps * 6), 3)
    dens = np.convolve(rate, np.ones(k, dtype=np.float32) / k, mode="same")
    hi = float(dens.max())
    return dens / hi if hi > 1e-6 else dens


def _voiceness(tl: Timeline) -> np.ndarray:
    """0..1 "somebody is speaking here".

    The subtitle track, where there is one, is a human being's answer to that
    question and beats any spectral guess. It is still blended rather than
    used alone: a cue is on or off, and the audio underneath it says whether
    the line is muttered or screamed.
    """
    heard = np.clip(tl.voice * np.clip(tl.speech / 0.62, 0.0, 1.35), 0.0, 1.0)
    if tl.dialogue is None:
        return heard
    return np.clip(0.62 * tl.dialogue + 0.38 * heard, 0.0, 1.0)


def _archetypes(tl: Timeline, weight: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample scores for the two things worth clipping: fight, talk."""
    vis = np.clip(0.68 * tl.motion + 0.32 * tl.contrast, 0.0, 1.0)
    cuts = _cut_rate(tl)

    if tl.has_audio:
        # A theme song is loud, fast and busy; the thing that gives it away is
        # that the level never drops. Discount both archetypes by it so a
        # montage that slipped past `segments` cannot win on energy alone.
        calm = 1.0 - 0.55 * tl.bed
        fight = (0.42 * vis + 0.30 * tl.impact + 0.16 * tl.loudness
                 + 0.12 * cuts) * calm
        # Loud on its own is an explosion in an empty room. Delivery is the
        # dynamics: a line rising into a shout modulates, a monologue does not.
        heat = 0.34 + 0.36 * tl.dynamics + 0.30 * tl.loudness
        talk = _voiceness(tl) * heat * calm
    else:
        fight = 0.72 * vis + 0.28 * cuts
        talk = np.zeros_like(fight)

    # Black frames, fades and flat title cards are never the clip.
    dark = tl.brightness < max(0.02, float(np.median(tl.brightness)) * 0.3)
    kill = np.where(dark | (tl.contrast < 0.03), 0.05, 1.0).astype(np.float32)

    # Smooth so a single loud frame cannot win a window on its own.
    k = max(int(tl.fps * 1.5), 3)
    kern = np.ones(k, dtype=np.float32) / k
    out = []
    for a in (fight, talk):
        a = np.convolve(a * kill * weight, kern, mode="same")
        # Normalise each archetype against its own spread over the usable part
        # of the file, so "the best fight" and "the best conversation" land on
        # the same scale and a talky episode is not scored as though nothing
        # happened in it.
        live = a[weight > 0.5]
        hi = float(np.percentile(live if len(live) > 8 else a, 97.0))
        out.append(a / hi if hi > 1e-6 else a)
    return np.clip(out[0], 0.0, 1.4), np.clip(out[1], 0.0, 1.4)


def _windows(sig: np.ndarray, win: int) -> np.ndarray:
    """Score of a clip-length window starting at each sample.

    Mostly the mean — a clip is however many seconds long and all of them have
    to hold up — but with a share from the peak, which is what separates a
    scene with a moment in it from one that is evenly mildly interesting.
    """
    cum = np.concatenate([[0.0], np.cumsum(sig)])
    mean = (cum[win:] - cum[:-win]) / win
    view = np.lib.stride_tricks.sliding_window_view(sig, win)
    return (0.76 * mean + 0.24 * view.max(axis=-1)).astype(np.float32)


def select(
    tl: Timeline,
    *,
    count: int,
    length: float,
    structure: Structure | None = None,
    skip_intro: float = 0.0,
    skip_outro: float = 0.0,
    min_gap: float | None = None,
) -> list[Clip]:
    """Choose `count` non-overlapping windows of `length` seconds.

    Greedy on window score with an enforced gap, which spreads picks across the
    runtime. Taking the literal top N instead would hand back five clips of the
    same two-minute battle. Where there is enough of both, the batch is held to
    a mix of fights and conversations rather than whichever the episode has
    more of.
    """
    n = len(tl.brightness)
    weight = (structure.weight if structure is not None and structure.weight is not None
              else np.ones(n, dtype=np.float32))
    fight, talk = _archetypes(tl, weight)
    fps = tl.fps
    win = max(int(round(length * fps)), 2)

    if win >= n:
        return [Clip(index=0, start=0.0, end=min(length, tl.duration), score=1.0)]

    wf, wt = _windows(fight, win), _windows(talk, win)
    scores = np.maximum(wf, wt)
    kinds = np.where(wf >= wt, "fight", "talk")

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
    # Never pick a window that reaches into a theme or a preview, however it
    # scored. A clip that runs three seconds into the credits is not a clip
    # that is 90% fine, so this is a share of the window rather than a mean of
    # the weights: a montage stays eligible, a theme does not.
    allowed = np.cumsum(np.concatenate([[0.0], (weight > 0.05).astype(np.float64)]))
    live = (allowed[win:] - allowed[:-win]) / win >= 0.98

    picks: list[int] = []
    cap = int(np.ceil(count * 0.65)) if count >= 3 else count
    taken = {"fight": 0, "talk": 0}

    def sweep(gap_frames: int, quota: bool) -> None:
        for i in order:
            if len(picks) >= count:
                return
            i = int(i)
            if not mask[i] or not live[i] or i in picks:
                continue
            if any(abs(i - p) < gap_frames for p in picks):
                continue
            if quota and taken[kinds[i]] >= cap:
                continue
            picks.append(i)
            taken[kinds[i]] += 1

    sweep(gap, quota=True)
    sweep(gap, quota=False)          # one-sided episode: fill with what it has
    if len(picks) < count:
        # Source too short (or too uniform) to honour the spacing — relax it
        # rather than return fewer clips than asked for.
        sweep(int(length * fps * 1.02), quota=False)
    if len(picks) < count:
        live = np.ones(len(scores), dtype=bool)   # last resort: ignore structure
        sweep(int(length * fps * 1.02), quota=False)

    picks.sort()
    return [
        Clip(index=k, start=p / fps, end=p / fps + length,
             score=float(scores[p]), kind=str(kinds[p]))
        for k, p in enumerate(picks)
    ]


# --------------------------------------------------------------------------
# alignment and framing
# --------------------------------------------------------------------------

def snap_and_frame(src: Path, tl: Timeline, clips: list[Clip],
                   *, snap: float = 1.4) -> list[Clip]:
    """Slide each clip onto shot boundaries and work out its framing.

    A clip that opens three frames into a shot reads as a mistake, and one that
    stops three frames before the next cut reads as a dropped connection. Both
    ends are graded, and the whole window slides — length stays exactly what
    was asked for.
    """
    total = tl.duration
    for c in clips:
        cuts = refine_cuts(src, c.start, c.end)
        want = c.duration
        if cuts:
            best = _best_shift(c, cuts, tl, snap)
            if best is not None:
                c.start = max(0.0, min(best, max(total - want, 0.0)))
                c.end = c.start + want

        inner = [t for t in cuts if c.start + 0.25 < t < c.end - 0.25]
        c.framing = _framing(tl, c, inner)
    return clips


def _best_shift(clip: Clip, cuts: list[float], tl: Timeline,
                snap: float) -> float | None:
    """Pick the start time, near the chosen one, that lands both ends well."""
    want = clip.duration
    ends = list(cuts) + [a for a, _ in tl.quiet]

    def grade(start: float) -> float:
        end = start + want
        near_end = min((abs(t - end) for t in ends), default=9.9)
        score = 1.0 - min(near_end / 1.6, 1.0)
        # Opening on black is worse than opening a beat late.
        if tl.brightness[tl.index(start)] < 0.05:
            score -= 1.5
        return score - abs(start - clip.start) / max(snap, 1e-6) * 0.30

    cands = [t for t in cuts if abs(t - clip.start) <= snap]
    if not cands:
        return None
    cands.append(clip.start)
    return max(cands, key=grade)


def _framing(tl: Timeline, clip: Clip, cuts: list[float]) -> list[tuple[float, float]]:
    """Crop position per shot: (offset from clip start, position 0..1).

    Held constant within a shot and stepped at cuts — the step is invisible
    because it happens on a cut, whereas a continuously moving crop reads as a
    drifting, unmotivated camera move. Only `fill` framing uses this.
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


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def _even(x: float) -> int:
    return max(int(round(x / 2.0)) * 2, 2)


def resolve_frame(mode: str, src_w: int, src_h: int, aspect: str) -> str:
    """Turn `auto` into a real framing choice for this source and shape."""
    if mode in ("fit", "fill", "pad"):
        return mode
    ow, oh = ASPECTS.get(aspect, ASPECTS["9:16"])
    if src_w <= 0 or src_h <= 0:
        return "fit"
    src_ar, out_ar = src_w / src_h, ow / oh
    keep = min(src_ar, out_ar) / max(src_ar, out_ar)
    return "fill" if keep >= _FILL_FLOOR else "fit"


def _scale_for(src_w: int, src_h: int, ow: int, oh: int, zoom: float) -> float:
    """Scale factor for the picture, honouring `zoom` but never past a full crop.

    Zooming beyond the point where the picture already covers the canvas buys
    nothing and only throws frame away, so a source that is already the right
    shape ignores `zoom` entirely rather than being cropped for no reason.
    """
    fit = min(ow / max(src_w, 1), oh / max(src_h, 1))
    src_ar, out_ar = max(src_w, 1) / max(src_h, 1), ow / oh
    full = max(src_ar / out_ar, out_ar / src_ar)     # zoom that fills the canvas
    return fit * min(max(zoom, 1.0), full)


def frame_note(mode: str, zoom: float, src_w: int, src_h: int,
               aspect: str) -> str:
    """One phrase describing what the viewer will actually see, for progress."""
    if mode == "fill":
        return "cropped to the action"
    ow, oh = ASPECTS.get(aspect, ASPECTS["9:16"])
    scale = _scale_for(src_w, src_h, ow, oh, zoom)
    kept = min(1.0, ow / max(src_w * scale, 1e-6))
    where = "on black" if mode == "pad" else "over a blurred backdrop"
    if kept > 0.995:
        return f"whole frame {where}"
    if kept < 0.4:
        return "cropped to the action"
    return f"{kept * 100:.0f}% of the width {where}"


def _video_chain(clip: Clip, *, frame: str, src_w: int, src_h: int,
                 aspect: str, sharpen: bool, fade: float,
                 zoom: float = 1.0) -> str:
    """The whole -vf graph for one clip."""
    ow, oh = ASPECTS.get(aspect, ASPECTS["9:16"])
    # Rebase timestamps to zero first. A fast seek lands the video on the next
    # whole frame but the audio on the next packet, which leaves the two
    # streams starting tens of milliseconds apart in the container — enough to
    # read as a lip-sync error. It also makes `t` in the filters below measure
    # from the clip start, which the framing expression and the fades rely on.
    head = "setpts=PTS-STARTPTS"
    crisp = "unsharp=5:5:0.5:5:5:0.0" if sharpen else None

    if frame != "fill":
        # Zoomed far enough that the picture covers the canvas on its own,
        # which is what `fill` is. Take that path rather than blurring a
        # backdrop no one will ever see.
        covers = _scale_for(src_w, src_h, ow, oh, zoom)
        if src_w * covers >= ow - 1 and src_h * covers >= oh - 1:
            frame = "fill"

    if frame == "fill":
        chain = [
            head,
            f"scale={ow}:{oh}:force_original_aspect_ratio=increase:flags=lanczos",
            f"crop={ow}:{oh}:x='{_crop_x(clip.framing)}':y=(ih-oh)/2",
            "setsar=1",
        ]
        # Cropping 16:9 into 9:16 means upscaling ~1.8x; a light luma sharpen
        # restores the edge definition that costs, without ringing.
        if crisp:
            chain.append(crisp)
        graph = ",".join(chain)
    else:
        # `zoom` is a dial between the two extremes rather than a third mode:
        # 1.0 is the whole frame, and enough of it fills the canvas and becomes
        # `fill`. In between, the picture is larger and only the outer edges of
        # the shot are lost — which is the trade most vertical clips actually
        # want, since the whole frame in a 9:16 canvas is a third of its height.
        scale = _scale_for(src_w, src_h, ow, oh, zoom)
        fw, fh = _even(src_w * scale), _even(src_h * scale)
        # What survives after trimming whatever overflows the canvas.
        vw, vh = min(fw, ow), min(fh, oh)
        # Overlay offsets have to be even or the chroma planes land half a
        # sample out and the whole picture picks up a colour fringe.
        x, y = min(_even((ow - vw) / 2), ow - vw), min(_even((oh - vh) / 2), oh - vh)

        fg = [f"scale={fw}:{fh}:flags=lanczos"]
        if (fw, fh) != (vw, vh):
            # Trimming the sides is a crop, so it may as well be the crop that
            # follows the action — the same per-shot position `fill` uses.
            fg.append(f"crop={vw}:{vh}:x='{_crop_x(clip.framing)}':y=(ih-oh)/2")
        if crisp and scale > 1.02:
            fg.append(crisp)
        fg.append("setsar=1")

        if frame == "pad":
            graph = (f"{head},{','.join(fg)},"
                     f"pad={ow}:{oh}:{x}:{y}:color=black")
        else:
            # The backdrop is the same frame, blown up to fill and blurred into
            # abstraction. Blurring at full size costs more than the encode
            # does, so it is done at a sixth of the resolution and scaled back
            # up — at this radius the two are indistinguishable.
            bw, bh = _even(ow / 6), _even(oh / 6)
            bg = (f"scale={bw}:{bh}:force_original_aspect_ratio=increase,"
                  f"crop={bw}:{bh},gblur=sigma=7:steps=2,"
                  f"eq=brightness=-0.09:saturation=1.45,"
                  f"scale={ow}:{oh}:flags=bilinear,setsar=1")
            graph = (f"{head},split=2[bgsrc][fgsrc];"
                     f"[bgsrc]{bg}[bg];"
                     f"[fgsrc]{','.join(fg)}[fg];"
                     f"[bg][fg]overlay={x}:{y}:format=auto")

    tail = ["format=yuv420p",
            # Tag the colour space on the frames themselves. Passing
            # -colorspace and friends as encoder options only lands the matrix
            # in this ffmpeg build; untagged primaries/transfer are why a
            # re-uploaded clip can come back looking washed out.
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"]
    if fade > 0:
        tail.append(f"fade=t=in:st=0:d={fade:.2f}")
        tail.append(f"fade=t=out:st={max(clip.duration - fade, 0):.3f}:d={fade:.2f}")
    return graph + "," + ",".join(tail)


def render_clip(
    src: Path, clip: Clip, dst: Path, *,
    aspect: str = "9:16",
    frame: str = "auto",
    zoom: float = 1.0,
    src_size: tuple[int, int] | None = None,
    crf: int = 19,
    sharpen: bool = True,
    has_audio: bool = True,
    normalize_audio: bool = True,
    fade: float = 0.12,
) -> Path:
    """Cut and encode one clip, audio in sync, ready to upload."""
    if src_size is None:
        info = probe(src)
        src_size = (info.width, info.height)
    src_w, src_h = src_size
    mode = resolve_frame(frame, src_w, src_h, aspect)

    vf = _video_chain(clip, frame=mode, src_w=src_w, src_h=src_h,
                      aspect=aspect, sharpen=sharpen, fade=fade, zoom=zoom)

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
        "-vf", vf, *maps,
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
