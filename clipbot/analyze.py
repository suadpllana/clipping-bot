"""Whole-file coarse analysis.

One low-resolution decode pass over the source produces everything clip
selection needs: shot boundaries, per-moment visual interest, where in the
frame the action actually sits, and how loud/dynamic the audio is.

The pass is deliberately cheap — 64x36 greyscale at a few samples per second —
because the source may be a two-hour film. Precision where it matters (the
exact frame a clip starts on) is recovered later by `refine_cuts`, which only
looks at the handful of windows that were actually chosen.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg import ffmpeg_bin, probe

GW, GH = 64, 36          # analysis grid
_FRAME = GW * GH
_ASR = 4000              # audio analysis sample rate


@dataclass
class Timeline:
    """Per-sample signals over the whole source, on a uniform time grid."""
    fps: float                    # samples per second
    duration: float
    brightness: np.ndarray        # 0..1 mean luma
    contrast: np.ndarray          # 0..1 luma spread
    motion: np.ndarray            # 0..1 frame-to-frame change
    centroid: np.ndarray          # 0..1 horizontal centre of activity
    cuts: np.ndarray              # shot boundary times, seconds
    loudness: np.ndarray          # 0..1 audio RMS, zeros if no audio
    dynamics: np.ndarray          # 0..1 local variation in loudness
    has_audio: bool

    def t(self, i: int) -> float:
        return i / self.fps

    def index(self, t: float) -> int:
        return int(np.clip(round(t * self.fps), 0, len(self.brightness) - 1))


def _decode_grid(src: Path, fps: float, *, hwaccel: bool,
                 on_progress=None, expect: int = 0) -> np.ndarray:
    """Decode the whole file to a stack of tiny greyscale frames."""
    pre: list[str] = []
    if hwaccel:
        # videotoolbox roughly halves decode time on Apple silicon. It is not
        # available everywhere and fails on some codecs, hence the retry path
        # in `scan`.
        pre = ["-hwaccel", "videotoolbox"]

    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
        *pre, "-i", str(src), "-an", "-sn", "-dn",
        "-vf", f"fps={fps},scale={GW}:{GH},format=gray",
        "-f", "rawvideo", "-",
    ]

    frames: list[np.ndarray] = []
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        assert proc.stdout is not None
        buf = b""
        try:
            while True:
                chunk = proc.stdout.read(_FRAME * 64)
                if not chunk:
                    break
                buf += chunk
                n = len(buf) // _FRAME
                if n:
                    block = np.frombuffer(buf[:n * _FRAME], dtype=np.uint8)
                    frames.append(block.reshape(n, GH, GW))
                    buf = buf[n * _FRAME:]
                    if on_progress and expect:
                        seen = sum(len(f) for f in frames)
                        on_progress(min(seen / expect, 1.0))
        finally:
            proc.stdout.close()
            proc.wait()

        if proc.returncode != 0 and not frames:
            err.seek(0)
            tail = err.read().decode("utf-8", "replace").strip().splitlines()[-6:]
            raise RuntimeError("ffmpeg could not decode the video:\n" + "\n".join(tail))

    if not frames:
        raise RuntimeError("No frames decoded — the file may be corrupt.")
    return np.concatenate(frames, axis=0)


def _audio_envelope(src: Path, fps: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """RMS loudness and local dynamics, resampled onto the video grid."""
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(src), "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(_ASR), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) < _ASR:
        return np.zeros(n), np.zeros(n)

    y = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    hop = max(int(_ASR / fps), 1)
    usable = (len(y) // hop) * hop
    if usable < hop:
        return np.zeros(n), np.zeros(n)
    win = y[:usable].reshape(-1, hop)
    rms = np.sqrt((win ** 2).mean(axis=1))

    # Perceptual-ish: loudness is logarithmic, and a linear RMS makes quiet
    # dialogue look like silence next to one gunshot.
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    db = np.clip(db, -60.0, 0.0)
    loud = (db + 60.0) / 60.0

    # Dialogue and action modulate; room tone and score drones do not. Local
    # spread separates "something is happening" from "something is playing".
    k = max(int(fps * 1.5), 3)
    pad = np.pad(loud, (k // 2, k // 2), mode="edge")
    strides = np.lib.stride_tricks.sliding_window_view(pad, k)[:len(loud)]
    dyn = strides.std(axis=1)
    hi = float(dyn.max())
    dyn = dyn / hi if hi > 1e-6 else dyn

    return _fit(loud, n), _fit(dyn, n)


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    """Resample a 1-D signal to exactly n samples."""
    if len(a) == n:
        return a
    if len(a) < 2:
        return np.full(n, float(a[0]) if len(a) else 0.0)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)


def _detect_cuts(frames: np.ndarray, fps: float) -> np.ndarray:
    """Shot boundaries from frame-to-frame difference.

    The threshold is derived from the file's own distribution rather than fixed:
    an animated episode with flat cels and a grainy live-action transfer sit at
    very different baseline diffs, and a constant misses every cut in one of
    them.
    """
    if len(frames) < 3:
        return np.array([0.0])

    f = frames.astype(np.float32)
    diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med))) or 1.0
    thresh = max(med + 6.0 * mad, 8.0)

    idx = np.flatnonzero(diff > thresh) + 1
    times = idx / fps

    # Collapse runs (a dissolve trips several consecutive samples).
    keep: list[float] = [0.0]
    for t in times:
        if t - keep[-1] > 0.35:
            keep.append(float(t))
    return np.array(keep)


def _centroid(frames: np.ndarray) -> np.ndarray:
    """Where the action is, horizontally, as 0 (left) .. 1 (right).

    A centre crop of 16:9 throws away 60% of the frame width, so a subject
    standing off-centre gets sliced in half. Column activity — motion plus
    detail — is a cheap stand-in for "what the viewer is looking at".
    """
    f = frames.astype(np.float32)
    detail = f.std(axis=1)                                  # (t, W)
    move = np.zeros_like(detail)
    move[1:] = np.abs(np.diff(f, axis=0)).mean(axis=1)

    weight = detail + 3.0 * move
    # Ignore the flat letterbox/pillarbox columns that would drag the centroid.
    weight = np.maximum(weight - weight.mean(axis=1, keepdims=True) * 0.5, 0.0)

    cols = np.arange(frames.shape[2], dtype=np.float32)
    total = weight.sum(axis=1)
    cen = np.where(total > 1e-6, (weight * cols).sum(axis=1) / np.maximum(total, 1e-6),
                   (frames.shape[2] - 1) / 2.0)
    return cen / max(frames.shape[2] - 1, 1)


def scan(src: Path, *, fps: float = 4.0, on_progress=None) -> Timeline:
    """Run the coarse pass over `src`."""
    info = probe(src)
    expect = int(info.duration * fps) if info.duration else 0

    try:
        frames = _decode_grid(src, fps, hwaccel=True,
                              on_progress=on_progress, expect=expect)
    except RuntimeError:
        # Hardware decode is a best-effort speedup; software always works.
        frames = _decode_grid(src, fps, hwaccel=False,
                              on_progress=on_progress, expect=expect)

    n = len(frames)
    f = frames.astype(np.float32)
    brightness = f.mean(axis=(1, 2)) / 255.0
    contrast = f.std(axis=(1, 2)) / 128.0
    motion = np.zeros(n, dtype=np.float32)
    motion[1:] = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2)) / 255.0

    loud, dyn = (_audio_envelope(src, fps, n) if info.has_audio
                 else (np.zeros(n), np.zeros(n)))

    return Timeline(
        fps=fps,
        duration=n / fps,
        brightness=brightness,
        contrast=np.clip(contrast, 0.0, 1.0),
        motion=np.clip(motion * 8.0, 0.0, 1.0),
        centroid=_centroid(frames),
        cuts=_detect_cuts(frames, fps),
        loudness=loud,
        dynamics=dyn,
        has_audio=info.has_audio,
    )


def refine_cuts(src: Path, start: float, end: float,
                *, fps: float = 24.0) -> list[float]:
    """Frame-accurate shot boundaries inside one short window.

    Called only for the windows that were actually selected, so the cost is a
    few seconds of video rather than the whole film.
    """
    pad = 1.0
    a = max(start - pad, 0.0)
    dur = (end + pad) - a
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-accurate_seek", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}",
        "-i", str(src), "-an", "-sn", "-dn",
        "-vf", f"fps={fps},scale={GW}:{GH},format=gray",
        "-f", "rawvideo", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    usable = (len(proc.stdout) // _FRAME) * _FRAME
    if usable < _FRAME * 3:
        return []
    frames = np.frombuffer(proc.stdout[:usable], dtype=np.uint8).reshape(-1, GH, GW)
    return [a + t for t in _detect_cuts(frames, fps).tolist()]
