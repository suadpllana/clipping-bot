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

# Homebrew does not always put its bin on PATH for GUI-launched processes.
_EXTRA_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")

_INSTALL_HINT = (
    "Install it with:\n"
    "  macOS    brew install ffmpeg\n"
    "  Windows  winget install Gyan.FFmpeg\n"
    "  Linux    sudo apt install ffmpeg"
)

_cache: dict[str, str] = {}


def _resolve(name: str) -> str:
    """Find a binary on PATH, falling back to known install locations.

    Package managers edit PATH for *new* shells, so a freshly-installed ffmpeg
    is often invisible to the process that installed it. Look in the usual
    package directories too.

    Resolution is lazy and cached rather than done at import: the web server
    must be able to start and report a readable error instead of dying with an
    ImportError traceback before it can serve a page.
    """
    if name in _cache:
        return _cache[name]

    found = shutil.which(name)
    if not found:
        for d in _EXTRA_DIRS:
            cand = Path(d) / name
            if cand.is_file() and os.access(cand, os.X_OK):
                found = str(cand)
                break
    if not found and _WINGET_BIN.exists():
        for exe in _WINGET_BIN.glob(f"*/bin/{name}.exe"):
            found = str(exe)
            break
    if not found:
        raise RuntimeError(f"{name} not found. {_INSTALL_HINT}")

    _cache[name] = found
    return found


def ffmpeg_bin() -> str:
    return _resolve("ffmpeg")


def ffprobe_bin() -> str:
    return _resolve("ffprobe")


def missing_tools() -> list[str]:
    """Which of the required binaries are absent. Empty list means good to go."""
    out = []
    for name in ("ffmpeg", "ffprobe"):
        try:
            _resolve(name)
        except RuntimeError:
            out.append(name)
    return out




@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def probe(path: Path) -> MediaInfo:
    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json",
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

    # Phones and some cameras tag rotation in metadata rather than baking it in;
    # ffprobe reports the stored dimensions, so 1080x1920 portrait footage looks
    # like landscape here and gets cropped the wrong way round.
    if _rotated_quarter_turn(video):
        return MediaInfo(duration, int(video["height"]), int(video["width"]),
                         fps or 30.0, has_audio)

    return MediaInfo(
        duration=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps or 30.0,
        has_audio=has_audio,
    )


def _rotated_quarter_turn(video: dict) -> bool:
    rot = 0.0
    for sd in video.get("side_data_list") or []:
        if "rotation" in sd:
            rot = float(sd["rotation"])
    tag = (video.get("tags") or {}).get("rotate")
    if tag:
        try:
            rot = float(tag)
        except ValueError:
            pass
    return abs(int(round(rot)) % 180) == 90


def run(args: list[str], *, quiet: bool = True) -> None:
    """Run ffmpeg, raising with real stderr on failure."""
    proc = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
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
