"""In-process job queue.

Rendering a full episode takes minutes, so the HTTP request that starts a job
cannot be the one that finishes it. Jobs run on worker threads and the UI polls
for progress.

Concurrency defaults to one. These are large-frame filter graphs — several at
once is how a slow render becomes an out-of-memory crash — so the limit is a
real constraint, not a knob to turn up casually.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pipeline import Cancelled, ClipResult, Settings, run_pipeline

QUEUED, RUNNING, DONE, ERROR, CANCELLED = (
    "queued", "running", "done", "error", "cancelled")


@dataclass
class Job:
    id: str
    settings: Settings
    source_name: str = ""
    state: str = QUEUED
    stage: str = "queued"
    message: str = "waiting for a free worker"
    progress: float = 0.0
    clips: list[ClipResult] = field(default_factory=list)
    error: str = ""
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    @property
    def eta(self) -> float | None:
        """Seconds remaining, extrapolated from progress so far.

        Rough by nature — the scan and the render advance at different rates —
        but a moving number is worth a lot next to a bar that just sits there.
        """
        if self.state != RUNNING or self.progress < 0.04:
            return None
        return max((self.elapsed / self.progress) - self.elapsed, 0.0)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "stage": self.stage,
            "message": self.message,
            "progress": round(self.progress, 4),
            "error": self.error,
            "elapsed": round(self.elapsed, 1),
            "eta": round(self.eta, 1) if self.eta is not None else None,
            "source": self.source_name,
            "mode": self.settings.mode,
            "aspect": self.settings.aspect,
            "requested": self.settings.count,
            "length": self.settings.length,
            "clips": [
                {
                    "index": c.index,
                    "start": round(c.start, 2),
                    "end": round(c.end, 2),
                    "duration": round(c.duration, 2),
                    "score": round(c.score, 4),
                    "size": c.size,
                    "name": c.path.name,
                    "tags": c.tags,
                    "thumb": bool(c.thumb),
                }
                for c in self.clips
            ],
        }


class JobStore:
    def __init__(self, workers: int = 1) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._q: queue.Queue[str] = queue.Queue()
        self._threads = [
            threading.Thread(target=self._worker, daemon=True,
                             name=f"clipbot-worker-{i}")
            for i in range(max(workers, 1))
        ]
        for t in self._threads:
            t.start()

    # -- api ---------------------------------------------------------------

    def submit(self, settings: Settings, source_name: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex[:12], settings=settings,
                  source_name=source_name)
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created,
                          reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.state in (DONE, ERROR, CANCELLED):
            return False
        job._cancel.set()
        if job.state == QUEUED:
            # Never picked up, so nothing will notice the flag — retire it here.
            job.state = CANCELLED
            job.message = "cancelled before it started"
            job.finished = time.time()
        else:
            job.message = "stopping after the current step…"
        return True

    def forget(self, job_id: str) -> bool:
        """Drop a finished job and delete everything it wrote."""
        job = self.get(job_id)
        if job is None:
            return False
        if job.state == RUNNING:
            job._cancel.set()
            return False
        with self._lock:
            self._jobs.pop(job_id, None)
        import shutil
        shutil.rmtree(job.settings.out_dir, ignore_errors=True)
        return True

    # -- worker ------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            job_id = self._q.get()
            job = self.get(job_id)
            if job is None or job._cancel.is_set():
                if job is not None and job.state == QUEUED:
                    job.state = CANCELLED
                    job.finished = time.time()
                self._q.task_done()
                continue
            self._run(job)
            self._q.task_done()

    def _run(self, job: Job) -> None:
        job.state = RUNNING
        job.started = time.time()
        job.stage = "starting"
        job.message = "starting up"
        job.progress = 0.0

        def report(stage: str, frac: float, message: str) -> None:
            job.stage = stage
            job.progress = max(0.0, min(float(frac), 1.0))
            job.message = message

        try:
            job.clips = run_pipeline(job.settings, report, job._cancel.is_set)
            job.state = DONE
            job.progress = 1.0
            job.stage = "done"
            job.message = f"{len(job.clips)} clips ready"
        except Cancelled:
            job.state = CANCELLED
            job.stage = "cancelled"
            job.message = "cancelled"
        except ValueError as e:
            # Anything the user can actually act on: bad settings, no ffmpeg,
            # source too short. Report it plainly with no traceback.
            job.state = ERROR
            job.stage = "error"
            job.error = str(e)
            job.message = str(e)
        except Exception as e:
            job.state = ERROR
            job.stage = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.message = job.error
            traceback.print_exc()
        finally:
            job.finished = time.time()


def clip_path(job: Job, index: int) -> Path | None:
    for c in job.clips:
        if c.index == index:
            return c.path
    return None


def thumb_path(job: Job, index: int) -> Path | None:
    for c in job.clips:
        if c.index == index:
            return c.thumb
    return None
