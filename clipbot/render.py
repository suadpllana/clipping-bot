"""Segment rendering, effects, and final mux.

Every segment is rendered to a lossless-ish intermediate at a fixed
resolution/fps/pixel-format so the concat demuxer can join them without
re-encoding mismatches, and so only the final pass costs real quality.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .ffmpeg import run
from .plan import Segment

W, H, FPS = 1080, 1920, 60

# Zoom effects are built from crop+scale rather than zoompan. zoompan renders
# its intermediate at the *input* resolution and holds a large internal canvas;
# fed an already-upscaled 1080x1920 frame it allocates well over a gigabyte per
# process and can exhaust system memory. crop+scale produces the same on-screen
# motion at a small fraction of the cost.
_MAX_ZOOM = 1.25


def _fit_vertical(bias: float = 0.0) -> str:
    """Scale-and-crop any aspect ratio to a filled 1080x1920 frame.

    force_original_aspect_ratio=increase then crop: fills the frame with no
    pillarboxing, at the cost of the sides of a widescreen shot. That is the
    correct trade for this format — bars read as lazy on TikTok.

    `bias` shifts the crop window horizontally, -1.0 (hard left) to 1.0 (hard
    right), 0.0 being centred. On a 2.39:1 source only ~23% of the frame width
    survives, so a centred window frequently cuts an off-centre subject in half;
    this is the escape hatch.
    """
    b = max(-1.0, min(1.0, bias))
    x = "(iw-ow)/2" if b == 0.0 else f"(iw-ow)/2+(iw-ow)/2*{b:.4f}"
    return (
        f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=bicubic,"
        f"crop={W}:{H}:x='{x}':y=(ih-oh)/2,setsar=1"
    )


def _zoom(expr: str, dx: str = "0", dy: str = "0") -> str:
    """A zoom/pan step: crop a shrinking window, then scale it back to full size.

    `expr` is a zoom factor >= 1.0 in terms of the output frame counter `n`.
    Crop dimensions are forced even (the `floor(../2)*2`) because odd sizes are
    invalid for yuv420p and make the filter fail at runtime.
    """
    cw = f"floor({W}/({expr})/2)*2"
    ch = f"floor({H}/({expr})/2)*2"
    x = f"(iw-out_w)/2+({dx})"
    y = f"(ih-out_h)/2+({dy})"
    return (
        f"crop=w='{cw}':h='{ch}':x='{x}':y='{y}':exact=1,"
        f"scale={W}:{H}:flags=bicubic,setsar=1"
    )


def _effect_chain(seg: Segment, frames: int) -> list[str]:
    """Translate effect names into filter strings.

    Each entry is a *linear* filter fragment: exactly one input and one output,
    no labels. Multi-branch effects (rgbsplit) are expressed with the `{IN}` and
    `{OUT}` placeholders and their own private link labels — `build_graph` is
    responsible for stitching those in. Keeping the distinction explicit is what
    stops a labelled sub-graph from being comma-joined into a linear chain,
    which produces a malformed graph.

    Zoom motion is driven by the frame counter `n` so it tracks the output frame
    rate exactly and cannot drift over the segment.
    """
    out: list[str] = []
    n = max(frames, 2)

    for fx in seg.effects:
        if fx == "punch":
            # Hard zoom in on the transient, easing back out over the segment.
            out.append(_zoom(f"max(1.18-0.010*n,1.0)"))
        elif fx == "driftin":
            out.append(_zoom(f"min(1.0+{0.14 / n:.6f}*n,1.14)"))
        elif fx == "driftout":
            out.append(_zoom(f"max(1.14-{0.14 / n:.6f}*n,1.0)"))
        elif fx == "zoomout":
            out.append(_zoom(f"max(1.25-{0.25 / n:.6f}*n,1.0)"))
        elif fx == "shake":
            # Zoom in slightly first so the offset never exposes the frame edge.
            out.append(_zoom("1.12", dx="sin(n/1.7)*14", dy="cos(n/1.3)*14"))
        elif fx == "flash":
            # 3-frame white lift on the cut.
            out.append(
                f"eq=brightness='if(lt(n,3),0.45-0.15*n,0)':eval=frame"
            )
        elif fx == "rgbsplit":
            # Chromatic aberration: shift the red channel a few px off the
            # green/blue. Multi-branch, so it carries {IN}/{OUT} placeholders.
            out.append(
                "{IN}split=2[rs_a][rs_b];"
                "[rs_a]lutrgb=g=0:b=0,"
                "crop=iw-6:ih:6:0,pad=iw+6:ih:0:0[rs_r];"
                "[rs_b]lutrgb=r=0[rs_gb];"
                "[rs_r][rs_gb]blend=all_mode=addition{OUT}"
            )
    return out


def build_graph(fragments: list[str], *, src: str = "0:v", dst: str = "v") -> str:
    """Stitch filter fragments into one valid filter_complex string.

    Linear fragments are comma-joined into a run; a multi-branch fragment (one
    containing {IN}/{OUT}) breaks the run and gets its own labelled links.
    """
    parts: list[str] = []
    run_buf: list[str] = []
    cur = f"[{src}]"
    n = 0

    def flush(out_label: str) -> None:
        nonlocal run_buf, cur
        if run_buf:
            parts.append(f"{cur}{','.join(run_buf)}{out_label}")
            run_buf = []
            cur = out_label

    for frag in fragments:
        if "{IN}" in frag:
            if run_buf:                     # only spend a label if there is a run
                flush(f"[fx{n}]")
                n += 1
            out_label = f"[fx{n}]"
            n += 1
            parts.append(frag.replace("{IN}", cur).replace("{OUT}", out_label))
            cur = out_label
        else:
            run_buf.append(frag)

    if run_buf:
        parts.append(f"{cur}{','.join(run_buf)}[{dst}]")
    else:
        # Last fragment was multi-branch; alias its output to the sink name.
        parts.append(f"{cur}null[{dst}]")
    return ";".join(parts)


def render_segment(seg: Segment, dst: Path, *, index: int,
                   bias: float = 0.0) -> Path:
    """Render one planned segment to an intermediate file."""
    src_dur = seg.duration * seg.speed
    frames = int(round(seg.duration * FPS))

    chain = [_fit_vertical(bias)]
    chain += _effect_chain(seg, frames)
    if seg.speed != 1.0:
        chain.append(f"setpts={1.0 / seg.speed:.5f}*PTS")
    chain.append(f"fps={FPS}")
    chain.append("format=yuv420p")
    graph = build_graph(chain)

    run([
        # Thread counts are capped deliberately: filter and encoder threads each
        # hold their own frame buffers, and at 1080x1920 an uncapped ffmpeg can
        # take gigabytes. Segment work is short, so the speed cost is small.
        "-threads", "2", "-filter_threads", "2", "-filter_complex_threads", "1",
        # -ss before -i seeks fast; -accurate_seek keeps the landing frame exact.
        "-accurate_seek", "-ss", f"{seg.src_start:.3f}",
        "-t", f"{src_dur:.3f}", "-i", str(seg.src),
        "-an", "-sn", "-dn",
        "-filter_complex", graph, "-map", "[v]",
        "-frames:v", str(frames),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
        "-x264-params", "threads=2:lookahead-threads=1:sliced-threads=0",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-video_track_timescale", "60000",
        str(dst),
    ])
    return dst


def safe_jobs(requested: int) -> int:
    """Clamp parallelism to what this machine's free memory can actually take.

    Each concurrent render needs roughly 700 MB at 1080x1920. Overcommitting
    here does not merely slow things down — it can drive the whole system into
    swap and force a hard power-off, so the cap is enforced rather than advised.
    """
    jobs = max(1, requested)
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MS()
        st.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        # Leave 2 GB for the OS and whatever else the user is running.
        usable_gb = max(st.ullAvailPhys / 2**30 - 2.0, 0.0)
        allowed = max(1, int(usable_gb / 0.7))
    except Exception:
        allowed = 1                      # unknown memory: assume the worst

    allowed = min(allowed, (os.cpu_count() or 2))
    return max(1, min(jobs, allowed))


def render_all(
    segments: list[Segment], work: Path, *, jobs: int = 1,
    progress=None, bias: float = 0.0,
) -> list[Path]:
    """Render every segment, sequentially by default.

    Parallelism is opt-in and memory-clamped: these are large-frame filter
    graphs, and several at once is what turns a slow render into an OOM.
    """
    jobs = safe_jobs(jobs)
    work.mkdir(parents=True, exist_ok=True)
    paths: list[Path | None] = [None] * len(segments)
    done = 0

    def _one(i_seg):
        i, seg = i_seg
        return i, render_segment(seg, work / f"seg_{i:04d}.mp4", index=i,
                                 bias=bias)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, path in pool.map(_one, enumerate(segments)):
            paths[i] = path
            done += 1
            if progress:
                progress(done, len(segments))

    return [p for p in paths if p is not None]


def concat_and_mux(
    parts: list[Path], audio: Path, audio_start: float,
    duration: float, dst: Path, work: Path,
    *, fade_out: float = 0.35,
) -> Path:
    """Join the rendered segments and lay the music over them."""
    listing = work / "concat.txt"
    listing.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8"
    )

    fade_start = max(duration - fade_out, 0.0)
    afilter = (
        f"afade=t=in:st=0:d=0.06,"
        f"afade=t=out:st={fade_start:.3f}:d={fade_out},"
        f"loudnorm=I=-14:TP=-1.0:LRA=11"
    )
    vfilter = f"fade=t=out:st={fade_start:.3f}:d={fade_out}"

    run([
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-accurate_seek", "-ss", f"{audio_start:.3f}", "-i", str(audio),
        "-filter_complex", f"[0:v]{vfilter}[v];[1:a]{afilter}[a]",
        "-map", "[v]", "-map", "[a]",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
        "-x264-params", "keyint=120:min-keyint=60:scenecut=0",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", "-r", str(FPS),
        str(dst),
    ])
    return dst
