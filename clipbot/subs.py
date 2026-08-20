"""Dialogue timing from an embedded subtitle track, when the file has one.

Most episode rips carry a text subtitle stream, and it is by far the cheapest
and most reliable answer to "is anybody talking here" — better than any
spectral guess, because a human wrote it down. Where it exists it is used to
sharpen selection; where it does not, everything still works from the audio
alone. Nothing here is load-bearing.

Image subtitles (PGS, VobSub) are ignored: they would need OCR, and the point
of this module is that it costs milliseconds.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg import ffmpeg_bin, ffprobe_bin

# Streams we can turn into text with ffmpeg alone.
TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}

# A cue that is only a music glyph is the captioner telling us the score is
# playing and nobody is speaking — which is the opposite of dialogue, and
# exactly what sits under an opening theme.
_MUSIC = re.compile(r"^[\s♪♬♫〜~…\-–—.、。]*$")
_TAGS = re.compile(r"<[^>]{0,40}>|\{[^}]{0,60}\}")
_TIME = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def is_music(self) -> bool:
        return bool(_MUSIC.match(self.text))


def _pick_stream(src: Path) -> int | None:
    """Index (among subtitle streams) of the best text track, if any."""
    try:
        out = subprocess.run(
            [ffprobe_bin(), "-v", "error", "-print_format", "json",
             "-select_streams", "s", "-show_streams", str(src)],
            capture_output=True, text=True, timeout=60,
        ).stdout
        streams = json.loads(out or "{}").get("streams") or []
    except Exception:
        return None

    best, best_n = None, -1.0
    for n, s in enumerate(streams):
        if (s.get("codec_name") or "").lower() not in TEXT_CODECS:
            continue
        # Prefer the fullest track. A "forced" or signs-only track has a
        # handful of cues and would read as an episode of near-total silence.
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        try:
            count = float(tags.get("number_of_frames", 0))
        except (TypeError, ValueError):
            count = 0.0
        if (s.get("disposition") or {}).get("forced"):
            count *= 0.1
        if count > best_n:
            best, best_n = n, count
    return best


def read_cues(src: Path) -> list[Cue]:
    """Extract and parse the subtitle track. Empty list if there is not one."""
    idx = _pick_stream(src)
    if idx is None:
        return []

    with tempfile.TemporaryDirectory(prefix="clipbot_subs_") as tmp:
        dst = Path(tmp) / "subs.srt"
        try:
            subprocess.run(
                [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-y", "-i", str(src), "-map", f"0:s:{idx}", "-c:s", "srt",
                 str(dst)],
                capture_output=True, timeout=300, check=True,
            )
            raw = dst.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

    cues: list[Cue] = []
    for block in re.split(r"\r?\n\r?\n", raw):
        m = _TIME.search(block)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        body = block[m.end():].strip()
        text = _TAGS.sub("", body).replace("\\N", " ").strip()
        if end > start:
            cues.append(Cue(start, end, text))
    return cues


def dialogue_track(src: Path, fps: float, n: int) -> np.ndarray | None:
    """1 where somebody is speaking, 0 elsewhere. None if there is no usable track.

    "Usable" means the track covers a plausible share of the runtime. A
    signs-and-songs track over a dubbed episode would otherwise mark the whole
    thing as wordless and push selection away from every conversation in it.
    """
    try:
        cues = read_cues(src)
    except Exception:
        return None
    if len(cues) < 12 or n <= 0:
        return None

    track = np.zeros(n, dtype=np.float32)
    spoken = 0.0
    for c in cues:
        if c.is_music:
            continue
        a = int(np.clip(round(c.start * fps), 0, n - 1))
        b = int(np.clip(round(c.end * fps), 0, n))
        if b > a:
            track[a:b] = 1.0
            spoken += c.end - c.start

    if spoken < (n / fps) * 0.08:
        return None
    return track
