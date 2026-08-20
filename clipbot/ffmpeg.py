"""Thin wrappers over the ffmpeg/ffprobe binaries."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_WINGET_BIN = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Microsoft/WinGet/Packages"
    / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
)


def _resolve(name: str) -> str:
    """Find a binary on PATH, falling back to the winget install location.

    winget edits PATH for *new* shells, so a freshly-installed ffmpeg is often
    invisible to the process that installed it. Look in the package dir too.
    """
    found = shutil.which(name)
    if found:
        return found
    if _WINGET_BIN.exists():
        for exe in _WINGET_BIN.glob(f"*/bin/{name}.exe"):
            return str(exe)
    raise RuntimeError(
        f"{name} not found. Install it with:  winget install Gyan.FFmpeg"
    )


FFMPEG = _resolve("ffmpeg")
FFPROBE = _resolve("ffprobe")


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def probe(path: Path) -> MediaInfo:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)

    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])

    # avg_frame_rate is a "num/den" string; it is "0/0" for some containers.
    num, _, den = video.get("avg_frame_rate", "0/0").partition("/")
    fps = float(num) / float(den) if den and float(den) != 0 else 30.0

    duration = float(data["format"].get("duration") or video.get("duration") or 0.0)
    return MediaInfo(
        duration=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps or 30.0,
        has_audio=has_audio,
    )


def run(args: list[str], *, quiet: bool = True) -> None:
    """Run ffmpeg, raising with real stderr on failure."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed:\n  args: {' '.join(args[:24])}…\n{tail}")
    if not quiet and proc.stderr.strip():
        print(proc.stderr.strip())


def extract_audio(src: Path, dst: Path) -> Path:
    """Decode any input to mono 22.05k wav for analysis."""
    run(["-i", str(src), "-vn", "-ac", "1", "-ar", "22050",
         "-c:a", "pcm_s16le", str(dst)])
    return dst
