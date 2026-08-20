"""Whole-file coarse analysis.

One low-resolution decode pass over the source produces everything clip
selection needs: shot boundaries, per-moment visual interest, where in the
frame the action actually sits, and what the audio is *doing* — talking,
fighting, or playing a theme song over a montage.

The pass is deliberately cheap — 64x36 greyscale at a few samples per second,
and audio reduced to three numbers per frame — because the source may be a
two-hour film. Precision where it matters (the exact frame a clip starts on)
is recovered later by `refine_cuts`, which only looks at the handful of windows
that were actually chosen.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ffmpeg import ffmpeg_bin, probe

GW, GH = 64, 36          # analysis grid
_FRAME = GW * GH

# Audio is analysed at 16 kHz because the speech formants that separate
# "someone is talking" from "the score is playing" run up to ~3.4 kHz, and a
# 4 kHz decode cuts straight through them. 1024/256 gives 62.5 feature frames
# a second — fine enough to see individual syllables.
_ASR = 16000
_AWIN, _AHOP = 1024, 256
_AFPS = _ASR / _AHOP


@dataclass
class Audio:
    """Reduced audio features, on their own uniform grid."""
    fps: float
    db: np.ndarray            # dBFS per frame, floored at -70
    loudness: np.ndarray      # 0..1 rescaled db
    dynamics: np.ndarray      # 0..1 local variation in loudness
    speech: np.ndarray        # 0..1 share of energy in the 300-3400 Hz band
    voice: np.ndarray         # 0..1 syllable-rate (2-8 Hz) envelope modulation
    impact: np.ndarray        # 0..1 transient/onset strength
    bed: np.ndarray           # 0..1 "loud floor, no gaps" — a music bed
    quiet: list[tuple[float, float]]   # near-silent gaps, seconds


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
    speech: np.ndarray            # 0..1 speech-band dominance
    voice: np.ndarray             # 0..1 syllable-rate modulation
    impact: np.ndarray            # 0..1 transient strength
    bed: np.ndarray               # 0..1 sustained, gapless music
    quiet: list[tuple[float, float]] = field(default_factory=list)
    dialogue: np.ndarray | None = None   # 1 where a subtitle cue is speaking
    has_audio: bool = False

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


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def _spectra(src: Path) -> np.ndarray:
    """Stream the audio through an STFT, keeping only reduced per-frame rows.

    A two-hour film is half a gigabyte of samples at 16 kHz and several times
    that as a materialised spectrogram, so the transform runs a block at a
    time and everything but three scalars per frame is thrown away
    immediately. Memory stays flat regardless of runtime.
    """
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(src), "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(_ASR), "-f", "s16le", "-",
    ]
    freqs = np.fft.rfftfreq(_AWIN, 1.0 / _ASR)
    m_speech = (freqs >= 300.0) & (freqs < 3400.0)
    window = np.hanning(_AWIN).astype(np.float32)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    carry = np.zeros(0, dtype=np.float32)
    prev: np.ndarray | None = None      # last spectrum of the previous block
    out: list[np.ndarray] = []
    try:
        while True:
            raw = proc.stdout.read(_ASR * 2 * 30)      # ~30 s of audio
            if not raw:
                break
            block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            y = np.concatenate([carry, block]) if len(carry) else block
            count = (len(y) - _AWIN) // _AHOP + 1
            if count <= 0:
                carry = y
                continue
            idx = np.arange(_AWIN)[None, :] + _AHOP * np.arange(count)[:, None]
            frames = y[idx] * window
            carry = y[count * _AHOP:]

            spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
            total = spec.sum(axis=1) + 1e-9
            rms = np.sqrt((frames ** 2).mean(axis=1))

            # Spectral flux, rectified: only rises count. A transient is energy
            # appearing, and rectifying is what stops a decaying cymbal from
            # reading as a second hit.
            ref = np.vstack([prev if prev is not None else spec[:1], spec[:-1]])
            rise = np.maximum(spec - ref, 0.0)
            prev = spec[-1:]

            out.append(np.stack([
                rms,
                spec[:, m_speech].sum(axis=1) / total,
                rise.sum(axis=1),
            ]).T.astype(np.float32))
    finally:
        proc.stdout.close()
        proc.wait()

    return np.concatenate(out, axis=0) if out else np.zeros((0, 3), np.float32)


def _roll(a: np.ndarray, k: int, fn) -> np.ndarray:
    """Apply `fn` over a centred rolling window of `k` samples."""
    k = max(int(k) | 1, 3)
    pad = np.pad(a, (k // 2, k // 2), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(pad, k)[:len(a)]
    return fn(view, -1)


def _norm(a: np.ndarray, pct: float = 97.0) -> np.ndarray:
    """Scale to 0..1 against a high percentile rather than the maximum.

    One clipped frame should not compress the whole rest of the signal into
    the bottom of the range.
    """
    hi = float(np.percentile(a, pct)) if len(a) else 0.0
    return np.clip(a / hi, 0.0, 1.0) if hi > 1e-9 else np.zeros_like(a)


def _bed(db: np.ndarray) -> np.ndarray:
    """0..1 evidence that this stretch is a music bed rather than a scene.

    Themes, credits and montages are mastered flat: the level never drops away
    between phrases. Dialogue does the opposite — the gaps between lines are
    where the floor lives. So a window whose *quiet* tenth is still loud, and
    whose loud-to-quiet spread is small, is almost always music. That single
    test is what separates an opening sequence from the fight it is cut like.
    """
    if len(db) < int(_AFPS * 2):
        return np.zeros(len(db), dtype=np.float32)

    w = max(int(_AFPS * 8) | 1, 9)
    step = max(int(_AFPS * 0.5), 1)
    pad = np.pad(db, (w // 2, w // 2), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(pad, w)[:len(db):step]
    lo = np.percentile(view, 10, axis=-1)
    hi = np.percentile(view, 90, axis=-1)

    floor = np.clip((lo + 50.0) / 14.0, 0.0, 1.0)       # -50 dB → 0, -36 dB → 1
    flat = np.clip((13.0 - (hi - lo)) / 8.0, 0.0, 1.0)  # 13 dB → 0, 5 dB → 1
    ev = np.where(lo <= -62.0, np.nan, floor * flat)

    # A window that is simply silent says nothing either way; carrying it as a
    # zero would punch a hole through a theme at every fade it contains.
    known = np.flatnonzero(~np.isnan(ev))
    if not len(known):
        return np.zeros(len(db), dtype=np.float32)
    ev = np.interp(np.arange(len(ev)), known, ev[known])

    # Themes run for a minute or more, so smooth hard: a ten-second lull in the
    # middle of one should not split it into two.
    ev = _roll(ev, int(24.0 / 0.5) | 1, np.mean)
    return np.interp(np.arange(len(db)),
                     np.arange(len(ev)) * step, ev).astype(np.float32)


def _quiet_gaps(db: np.ndarray) -> list[tuple[float, float]]:
    """Near-silent stretches, in seconds.

    Broadcast animation hard-cuts its audio at structural joins: the recap, the
    opening, the eyecatch and the ending are each bracketed by a beat of near
    silence. Those beats are the tightest segment boundaries in the file — far
    more precise than anything the loudness curve alone gives.
    """
    if not len(db):
        return []
    thr = float(np.clip(np.median(db) - 20.0, -60.0, -38.0))
    gaps: list[tuple[float, float]] = []
    q = db < thr
    edges = np.flatnonzero(np.diff(np.concatenate([[False], q, [False]]).astype(np.int8)))
    for a, b in zip(edges[::2], edges[1::2]):
        if (b - a) / _AFPS >= 0.30:
            gaps.append((float(a) / _AFPS, float(b) / _AFPS))
    return gaps


def _audio(src: Path) -> Audio | None:
    feat = _spectra(src)
    if len(feat) < int(_AFPS * 2):
        return None

    rms, speech, flux = feat[:, 0], feat[:, 1], feat[:, 2]

    # Perceptual-ish: loudness is logarithmic, and a linear RMS makes quiet
    # dialogue look like silence next to one gunshot.
    db = np.clip(20.0 * np.log10(np.maximum(rms, 1e-6)), -70.0, 0.0)
    loud = (db + 70.0) / 70.0

    # Dialogue and action modulate; room tone and score drones do not. Local
    # spread separates "something is happening" from "something is playing".
    dyn = _norm(_roll(loud, int(_AFPS * 1.5), np.std))

    # Syllables land at 2-8 Hz. A difference of two moving averages is a cheap
    # band-pass around that band, and its local energy runs high for speech and
    # low for a held note, sustained noise or a drone.
    fast = _roll(db, int(_AFPS * 0.12), np.mean)
    slow = _roll(db, int(_AFPS * 0.55), np.mean)
    syl = _roll(np.abs(fast - slow), int(_AFPS * 2.5), np.mean)
    # Gate on level: the noise floor modulates too, and it is not talking.
    voice = _norm(syl) * np.clip((db + 55.0) / 15.0, 0.0, 1.0)

    return Audio(
        fps=_AFPS,
        db=db.astype(np.float32),
        loudness=loud.astype(np.float32),
        dynamics=dyn.astype(np.float32),
        speech=np.clip(speech, 0.0, 1.0).astype(np.float32),
        voice=voice.astype(np.float32),
        impact=_norm(_roll(_norm(flux, 99.0), int(_AFPS * 0.8), np.max)),
        bed=_bed(db),
        quiet=_quiet_gaps(db),
    )


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


def scan(src: Path, *, fps: float = 4.0, on_progress=None,
         use_subs: bool = True) -> Timeline:
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

    aud = _audio(src) if info.has_audio else None
    zero = np.zeros(n, dtype=np.float32)

    tl = Timeline(
        fps=fps,
        duration=n / fps,
        brightness=brightness,
        contrast=np.clip(contrast, 0.0, 1.0),
        motion=np.clip(motion * 8.0, 0.0, 1.0),
        centroid=_centroid(frames),
        cuts=_detect_cuts(frames, fps),
        loudness=_fit(aud.loudness, n) if aud else zero,
        dynamics=_fit(aud.dynamics, n) if aud else zero,
        speech=_fit(aud.speech, n) if aud else zero,
        voice=_fit(aud.voice, n) if aud else zero,
        impact=_fit(aud.impact, n) if aud else zero,
        bed=_fit(aud.bed, n) if aud else zero,
        quiet=aud.quiet if aud else [],
        has_audio=bool(aud),
    )

    if use_subs:
        from .subs import dialogue_track
        tl.dialogue = dialogue_track(src, fps, n)
    return tl


def refine_cuts(src: Path, start: float, end: float,
                *, fps: float = 24.0) -> list[float]:
    """Frame-accurate shot boundaries inside one short window.

    Called only for the windows that were actually selected, so the cost is a
    few seconds of video rather than the whole film. The window is padded
    wider than the clip because `snap_and_frame` slides the clip within it,
    and a boundary it slid onto has to have been scanned.
    """
    pad = 2.0
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
