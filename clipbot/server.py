"""HTTP API and static host for the clipbot UI.

Uploads arrive as a raw streaming body rather than multipart. A full film is
several gigabytes; multipart parsing would buffer and re-copy all of it for no
benefit, where a straight `request.stream()` to disk is flat in memory and
roughly disk-speed.
"""
from __future__ import annotations

import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .ffmpeg import missing_tools, probe
from .highlight import ASPECTS
from .jobs import DONE, RUNNING, JobStore, clip_path, thumb_path
from .pipeline import QUALITY, Settings

DATA = Path(os.environ.get("CLIPBOT_DATA", Path.home() / ".clipbot")).expanduser()
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "clips"
WEB = Path(__file__).parent / "web"

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts", ".mpg",
              ".mpeg", ".wmv", ".flv", ".m2ts"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

CHUNK = 4 * 1024 * 1024

app = FastAPI(title="clipbot", version=__version__)
store = JobStore(workers=int(os.environ.get("CLIPBOT_WORKERS", "1")))


def _safe_name(raw: str) -> str:
    """Reduce a client-supplied filename to something safe to join to a path.

    Only the basename survives, and only from a conservative character set —
    the name comes from the browser, so it is untrusted input that ends up in a
    filesystem path.
    """
    base = Path(raw.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" ._-")
    return base[:120] or "upload"


def _upload_dir(uid: str) -> Path:
    """Resolve an upload id to its directory, rejecting anything traversing."""
    if not re.fullmatch(r"[a-f0-9]{8,32}", uid or ""):
        raise HTTPException(400, "Bad upload id.")
    d = UPLOADS / uid
    if not d.is_dir():
        raise HTTPException(404, "Upload not found — it may have been cleared.")
    return d


def _stored_file(uid: str) -> Path:
    d = _upload_dir(uid)
    files = [p for p in d.iterdir() if p.is_file()]
    if not files:
        raise HTTPException(404, "Upload is empty.")
    return files[0]


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    miss = missing_tools()
    usage = shutil.disk_usage(DATA if DATA.exists() else Path.home())
    return {
        "version": __version__,
        "ready": not miss,
        "missing": miss,
        "aspects": list(ASPECTS),
        "qualities": list(QUALITY),
        "data_dir": str(DATA),
        "free_gb": round(usage.free / 2**30, 1),
    }


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------

@app.put("/api/upload")
async def upload(request: Request, name: str = "", kind: str = "video") -> JSONResponse:
    if kind not in ("video", "audio"):
        raise HTTPException(400, "kind must be 'video' or 'audio'.")

    filename = _safe_name(name or request.headers.get("x-filename", "upload"))
    ext = Path(filename).suffix.lower()
    allowed = VIDEO_EXTS if kind == "video" else AUDIO_EXTS
    if ext not in allowed:
        raise HTTPException(
            400,
            f"{ext or 'That file type'} is not supported. "
            f"Try: {', '.join(sorted(allowed))}",
        )

    uid = uuid.uuid4().hex[:16]
    dest_dir = UPLOADS / uid
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    size = 0
    try:
        with dest.open("wb") as fh:
            async for chunk in request.stream():
                if chunk:
                    fh.write(chunk)
                    size += len(chunk)
    except Exception:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(400, "Upload failed or was interrupted.")

    if size == 0:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(400, "Received an empty file.")

    info: dict = {"id": uid, "name": filename, "size": size, "kind": kind}
    try:
        m = probe(dest)
        info |= {
            "duration": round(m.duration, 2),
            "width": m.width,
            "height": m.height,
            "fps": round(m.fps, 3),
            "has_audio": m.has_audio,
        }
    except Exception as e:
        # For audio, probe() legitimately fails (no video stream) — that is fine.
        if kind == "video":
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(
                400, f"That file could not be read as video ({e})."
            ) from e
        info["duration"] = None

    return JSONResponse(info)


@app.delete("/api/upload/{uid}")
def drop_upload(uid: str) -> dict:
    shutil.rmtree(_upload_dir(uid), ignore_errors=True)
    return {"ok": True}


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

@app.post("/api/jobs")
async def create_job(request: Request) -> JSONResponse:
    body = await request.json()

    video = _stored_file(str(body.get("video_id", "")))
    audio = None
    if body.get("audio_id"):
        audio = _stored_file(str(body["audio_id"]))

    mode = str(body.get("mode", "highlight"))
    if mode not in ("highlight", "music"):
        raise HTTPException(400, "mode must be 'highlight' or 'music'.")

    try:
        count = int(body.get("count", 5))
        length = float(body.get("length", 30))
        skip_intro = max(float(body.get("skip_intro", 0) or 0), 0.0)
        skip_outro = max(float(body.get("skip_outro", 0) or 0), 0.0)
        seed = int(body.get("seed") or (int(time.time()) % 100000))
    except (TypeError, ValueError):
        raise HTTPException(400, "Clip count, length and trims must be numbers.")

    out_dir = OUTPUTS / uuid.uuid4().hex[:12]
    settings = Settings(
        video=video,
        out_dir=out_dir,
        count=count,
        length=length,
        mode=mode,
        audio=audio,
        aspect=str(body.get("aspect", "9:16")),
        quality=str(body.get("quality", "high")),
        skip_intro=skip_intro,
        skip_outro=skip_outro,
        sharpen=bool(body.get("sharpen", True)),
        normalize_audio=bool(body.get("normalize_audio", True)),
        seed=seed,
    )
    try:
        settings.validate()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    job = store.submit(settings, source_name=video.name)
    return JSONResponse(job.public(), status_code=202)


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": [j.public() for j in store.all()[:40]]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    return job.public()


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    if store.get(job_id) is None:
        raise HTTPException(404, "No such job.")
    return {"ok": store.cancel(job_id)}


@app.delete("/api/jobs/{job_id}")
def job_delete(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    if job.state == RUNNING:
        store.cancel(job_id)
        raise HTTPException(409, "Job is still running — cancelled it instead.")
    return {"ok": store.forget(job_id)}


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

def _clip_file(job_id: str, index: int) -> tuple:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    path = clip_path(job, index)
    if path is None or not path.exists():
        raise HTTPException(404, "No such clip.")
    return job, path


@app.get("/api/jobs/{job_id}/clips/{index}")
def clip_download(job_id: str, index: int, download: int = 0):
    job, path = _clip_file(job_id, index)
    stem = Path(job.source_name).stem[:48] or "clip"
    # FileResponse handles Range requests, which is what makes the inline
    # preview scrubbable rather than download-then-play.
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{stem}_clip{index:02d}.mp4" if download else None,
    )


@app.get("/api/jobs/{job_id}/thumbs/{index}")
def clip_thumb(job_id: str, index: int):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    path = thumb_path(job, index)
    if path is None or not path.exists():
        raise HTTPException(404, "No thumbnail.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/zip")
def clip_zip(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    if job.state != DONE or not job.clips:
        raise HTTPException(409, "Clips are not ready yet.")

    stem = Path(job.source_name).stem[:48] or "clips"
    bundle = job.settings.out_dir / f"{_safe_name(stem)}_clips.zip"
    newest = max((c.path.stat().st_mtime for c in job.clips if c.path.exists()),
                 default=0.0)
    if not bundle.exists() or bundle.stat().st_mtime < newest:
        # ZIP_STORED, not DEFLATE: h264 does not compress twice, and deflating
        # a few hundred megabytes to save nothing is a pointless stall.
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as z:
            for c in job.clips:
                if c.path.exists():
                    z.write(c.path, arcname=c.path.name)
    return FileResponse(bundle, media_type="application/zip",
                        filename=bundle.name)


# --------------------------------------------------------------------------
# static UI — mounted last so it cannot shadow /api
# --------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")


def serve(host: str = "127.0.0.1", port: int = 8000, *, open_browser: bool = True):
    import uvicorn

    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    miss = missing_tools()
    banner = "  !! ffmpeg missing — install it: brew install ffmpeg" if miss else ""
    print(f"\n  clipbot {__version__}  →  http://{host}:{port}")
    print(f"  data: {DATA}")
    if banner:
        print(banner)
    print()

    if open_browser:
        import threading
        import webbrowser
        threading.Timer(
            0.9, lambda: webbrowser.open(f"http://{host}:{port}")
        ).start()

    uvicorn.run(app, host=host, port=port, log_level="warning",
                timeout_keep_alive=75)
