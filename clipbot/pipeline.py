"""Orchestration shared by the web UI and the CLI.

Two ways to make a clip out of a long source:

* **highlight** — find the best moments and cut them straight, keeping the
  source's own audio. This is what "upload an episode, get clips" means.
* **music** — the original beat-synced montage: shots from across the source
  cut to a supplied track.

Both are driven from here so the UI and the command line cannot drift apart.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import highlight, segments
from .analyze import scan
from .ffmpeg import missing_tools, probe

Report = Callable[[str, float, str], None]   # stage, 0..1, human message


class Cancelled(Exception):
    """Raised out of the pipeline when the caller asks it to stop."""


# crf: lower is better quality and a bigger file.
QUALITY = {"max": 16, "high": 19, "balanced": 22, "small": 26}


@dataclass
class Settings:
    video: Path
    out_dir: Path
    count: int = 5
    length: float = 30.0
    mode: str = "highlight"          # highlight | music
    audio: Path | None = None        # required for music mode
    aspect: str = "9:16"
    frame: str = "auto"              # auto | fit | fill | pad
    zoom: float = 1.4                # how far into the frame `fit`/`pad` push
    quality: str = "high"
    auto_skip: bool = True           # find and avoid recap / OP / ED / preview
    skip_intro: float = 0.0
    skip_outro: float = 0.0
    sharpen: bool = True
    normalize_audio: bool = True
    seed: int = 1
    scan_fps: float = 4.0

    def validate(self) -> None:
        miss = missing_tools()
        if miss:
            raise ValueError(
                f"{' and '.join(miss)} not found on this machine. "
                "Install ffmpeg (macOS: brew install ffmpeg) and restart."
            )
        if not self.video.exists():
            raise ValueError(f"No such video: {self.video}")
        if self.mode == "music":
            if not self.audio or not self.audio.exists():
                raise ValueError("Beat-synced mode needs a music track.")
        if not 1 <= self.count <= 40:
            raise ValueError("Clip count must be between 1 and 40.")
        if not 3.0 <= self.length <= 180.0:
            raise ValueError("Clip length must be between 3 and 180 seconds.")
        if self.aspect not in highlight.ASPECTS:
            raise ValueError(f"Unknown aspect ratio: {self.aspect}")
        if self.frame not in highlight.FRAMES:
            raise ValueError(f"Unknown framing mode: {self.frame}")
        if not 1.0 <= self.zoom <= 4.0:
            raise ValueError("Zoom must be between 1.0 and 4.0.")


@dataclass
class ClipResult:
    index: int
    start: float
    end: float
    duration: float
    score: float
    path: Path
    thumb: Path | None = None
    size: int = 0
    tags: list[str] = field(default_factory=list)


def _stamp(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


def _noop(*_a) -> None:
    pass


def run_pipeline(
    s: Settings,
    report: Report = _noop,
    cancelled: Callable[[], bool] = lambda: False,
) -> list[ClipResult]:
    s.validate()
    s.out_dir.mkdir(parents=True, exist_ok=True)

    def check() -> None:
        if cancelled():
            raise Cancelled()

    if s.mode == "music":
        return _run_music(s, report, check)
    return _run_highlight(s, report, check)


# --------------------------------------------------------------------------
# highlight mode
# --------------------------------------------------------------------------

def _run_highlight(s: Settings, report: Report, check) -> list[ClipResult]:
    info = probe(s.video)
    report("probe", 0.02,
           f"{info.width}x{info.height}, {_stamp(info.duration)} long")

    usable = max(info.duration - s.skip_intro - s.skip_outro, 0.0)
    if usable < s.length:
        raise ValueError(
            f"Source is only {_stamp(usable)} of usable footage — "
            f"not enough for a {s.length:g}s clip."
        )

    # 1. one coarse pass over the whole file ------------------------------
    def scan_prog(frac: float) -> None:
        check()
        report("scan", 0.03 + 0.42 * frac, "watching the whole thing through")

    report("scan", 0.03, "watching the whole thing through")
    tl = scan(s.video, fps=s.scan_fps, on_progress=scan_prog)
    check()
    report("scan", 0.46, f"{len(tl.cuts)} shots detected")

    # 2. work out which parts are not the story ---------------------------
    st = segments.analyse(tl, enabled=s.auto_skip)
    skipped = st.summary()
    if skipped:
        how = "subtitles" if st.used_subs else "the audio"
        report("select", 0.47, f"skipping {skipped} (found from {how})")
    elif s.auto_skip:
        report("select", 0.47, "no recap or theme found — using the whole file")

    # 3. pick the moments -------------------------------------------------
    clips = highlight.select(
        tl, count=s.count, length=s.length, structure=st,
        skip_intro=s.skip_intro, skip_outro=s.skip_outro,
    )
    if not clips:
        raise ValueError("Could not find any usable moments in this video.")
    kinds = ", ".join(
        f"{sum(c.kind == k for c in clips)} {k}"
        for k in ("fight", "talk") if any(c.kind == k for c in clips)
    )
    report("select", 0.49, f"{len(clips)} moments picked" + (f" ({kinds})" if kinds else ""))

    # 4. align to shot boundaries and resolve framing ----------------------
    clips = highlight.snap_and_frame(s.video, tl, clips)
    check()
    mode = highlight.resolve_frame(s.frame, info.width, info.height, s.aspect)
    note = highlight.frame_note(mode, s.zoom, info.width, info.height, s.aspect)
    report("select", 0.52, f"aligned to shot boundaries — {note}")

    # 5. render -----------------------------------------------------------
    has_audio = tl.has_audio
    crf = QUALITY.get(s.quality, 19)
    out: list[ClipResult] = []
    span = 0.45 / max(len(clips), 1)

    for n, c in enumerate(clips):
        check()
        base = 0.52 + n * span
        report("render", base,
               f"rendering clip {n + 1} of {len(clips)} — {_stamp(c.start)}")
        name = f"clip_{n + 1:02d}_{_stamp(c.start)}.mp4"
        dst = s.out_dir / name
        highlight.render_clip(
            s.video, c, dst, aspect=s.aspect, frame=s.frame, zoom=s.zoom,
            src_size=(info.width, info.height), crf=crf, sharpen=s.sharpen,
            has_audio=has_audio, normalize_audio=s.normalize_audio,
        )
        thumb = highlight.make_thumb(dst, s.out_dir / f"{dst.stem}.jpg")
        out.append(ClipResult(
            index=n + 1, start=c.start, end=c.end, duration=c.duration,
            score=c.score, path=dst, thumb=thumb,
            size=dst.stat().st_size if dst.exists() else 0,
            tags=_tags(tl, c),
        ))

    report("done", 1.0, f"{len(out)} clips ready")
    return out


def _tags(tl, clip) -> list[str]:
    """Short labels describing why a moment was picked, for the results grid."""
    i, j = tl.index(clip.start), tl.index(clip.end)
    sl = slice(i, max(j, i + 1))
    tags: list[str] = [clip.kind] if clip.kind in ("fight", "talk") else []

    if float(tl.motion[sl].mean()) > 0.35 and "fight" not in tags:
        tags.append("action")
    if ("talk" not in tags and tl.dialogue is not None
            and float(tl.dialogue[sl].mean()) > 0.72):
        tags.append("talky")
    if tl.has_audio and float(tl.loudness[sl].mean()) > 0.62:
        tags.append("loud")
    if tl.has_audio and float(tl.bed[sl].mean()) > 0.45:
        tags.append("scored")
    cuts = int(((tl.cuts >= clip.start) & (tl.cuts < clip.end)).sum())
    if cuts >= max(int(clip.duration / 2.5), 3):
        tags.append("fast-cut")
    if float(tl.brightness[sl].mean()) < 0.18:
        tags.append("dark")
    return tags or ["steady"]


# --------------------------------------------------------------------------
# music mode — the original beat-synced montage, once per clip
# --------------------------------------------------------------------------

def _run_music(s: Settings, report: Report, check) -> list[ClipResult]:
    try:
        from . import beats, plan, render, shots
    except ImportError as e:      # librosa / scenedetect absent
        raise ValueError(
            "Beat-synced mode needs the analysis extras: "
            f"pip install -r requirements.txt  ({e})"
        ) from e

    work = s.out_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)

    try:
        report("audio", 0.03, "analysing the track")
        bm = beats.analyze(s.audio, work)
        check()
        windows = _music_windows(bm, s.length, s.count)
        report("audio", 0.10,
               f"{bm.tempo:.0f} BPM, {len(windows)} section(s) chosen")

        report("shots", 0.12, "detecting shots")
        sources = shots.find_videos(s.video)
        if not sources:
            raise ValueError("No video files found.")
        pool: list = []
        for src in sources:
            check()
            pool += shots.split_shots(src, skip_intro=s.skip_intro,
                                      skip_outro=s.skip_outro)
        if not pool:
            raise ValueError("No shots detected in the source.")
        report("shots", 0.30, f"{len(pool)} raw shots")

        report("rank", 0.32, "ranking shots")
        pool = shots.score_shots(pool, sample=True)
        if not pool:
            raise ValueError("Every shot was rejected as blank or too dark.")
        check()
        report("rank", 0.45, f"{len(pool)} usable shots")

        bands = _time_bands(pool, s.count)
        need = int(s.length * 2.2) + 8
        out: list[ClipResult] = []
        span = 0.5 / max(s.count, 1)

        for n in range(s.count):
            check()
            base = 0.48 + n * span
            report("render", base, f"building clip {n + 1} of {s.count}")

            band = shots.diversify(bands[n], need)
            if len(band) < need // 2:          # thin band: top up from the rest
                band += shots.diversify(
                    [sh for sh in pool if sh not in band], need - len(band))
            segs = plan.build_plan(bm, band, window=windows[n % len(windows)],
                                   seed=s.seed + n * 7919)
            total = sum(seg.duration for seg in segs)

            def prog(done: int, count: int, _base=base) -> None:
                check()
                report("render", _base + span * 0.85 * (done / max(count, 1)),
                       f"clip {n + 1}: cut {done}/{count}")

            parts = render.render_all(segs, work / f"seg{n}", jobs=1,
                                      progress=prog)
            name = f"clip_{n + 1:02d}_beatsync.mp4"
            dst = s.out_dir / name
            render.concat_and_mux(parts, s.audio, windows[n % len(windows)][0],
                                  total, dst, work)
            shutil.rmtree(work / f"seg{n}", ignore_errors=True)

            thumb = highlight.make_thumb(dst, s.out_dir / f"{dst.stem}.jpg")
            out.append(ClipResult(
                index=n + 1, start=windows[n % len(windows)][0],
                end=windows[n % len(windows)][0] + total, duration=total,
                score=1.0 - n * 0.01, path=dst, thumb=thumb,
                size=dst.stat().st_size if dst.exists() else 0,
                tags=["beat-synced", f"{bm.tempo:.0f} BPM"],
            ))

        report("done", 1.0, f"{len(out)} clips ready")
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _music_windows(bm, length: float, count: int) -> list[tuple[float, float]]:
    """Pick `count` distinct sections of the track, best first.

    Every clip cut to the same 30 seconds of the song would be the same clip.
    Scored like `beats.pick_window`, then taken greedily with no overlap.
    """
    import numpy as np

    if bm.duration <= length * 1.15:
        return [(0.0, min(bm.duration, length))]

    cands = bm.downbeats or bm.beats
    scored: list[tuple[float, float]] = []
    for start in cands:
        end = start + length
        if end > bm.duration:
            break
        idx = [i for i, b in enumerate(bm.beats) if start <= b < end]
        if not idx:
            continue
        sc = float(np.mean([bm.energy[i] for i in idx]))
        if bm.drop is not None and start <= bm.drop < end:
            sc += 0.35 * (1.0 - min((bm.drop - start) / length, 1.0))
        scored.append((sc, start))

    scored.sort(reverse=True)
    picked: list[float] = []
    for _, start in scored:
        if len(picked) >= count:
            break
        if any(abs(start - p) < length * 0.85 for p in picked):
            continue
        picked.append(start)

    if not picked:
        picked = [0.0]
    return [(p, p + length) for p in picked]


def _time_bands(pool: list, count: int) -> list[list]:
    """Split the shot pool into `count` bands by position in the source.

    Clip 1 then comes from the opening of the episode and clip N from the end,
    instead of every clip drawing from the same handful of top-scoring shots.
    """
    if count <= 1:
        return [list(pool)]
    by_time = sorted(pool, key=lambda sh: (str(sh.src), sh.start))
    n = len(by_time)
    size = max(n // count, 1)
    bands = [by_time[i * size:(i + 1) * size] for i in range(count)]
    if n > size * count:
        bands[-1] += by_time[size * count:]
    # A band with nothing in it falls back to the whole pool.
    return [b if b else list(pool) for b in bands]
