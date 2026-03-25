"""Beat-Synced Squares — FastAPI: upload video, queue render job, poll status, download result."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

# Repo root on path for `import main`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main import render_tracked_effect  # noqa: E402

from api.schemas import JobCreateResponse, JobOptions, JobStatusResponse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024
_work_override = (
    os.environ.get("BEAT_SYNC_WORK_DIR", "").strip()
    or os.environ.get("EFFECT_WORK_DIR", "").strip()  # legacy alias
)
WORK_ROOT = (
    Path(_work_override)
    if _work_override
    else Path(os.environ.get("TMPDIR", "/tmp")) / "beat-synced-squares-jobs"
)
ALLOW_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")

_jobs_lock = threading.Lock()
JOBS: dict[str, dict] = {}

app = FastAPI(
    title="Beat-Synced Squares",
    description="Upload a video; get back a beat-synced ORB + optical-flow overlay as MP4.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS if o.strip()] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _job_dir(job_id: str) -> Path:
    return WORK_ROOT / job_id


def _process_job_sync(job_id: str, video_in: Path, video_out: Path, kwargs: dict) -> None:
    with _jobs_lock:
        JOBS[job_id]["status"] = "processing"
    try:
        render_tracked_effect(video_in=video_in, video_out=video_out, **kwargs)
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        with _jobs_lock:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
        return
    with _jobs_lock:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["output_path"] = str(video_out)


async def _run_job_task(job_id: str, video_in: Path, video_out: Path, kwargs: dict) -> None:
    await run_in_threadpool(_process_job_sync, job_id, video_in, video_out, kwargs)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/jobs", response_model=JobCreateResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: str = Form("{}"),
) -> JobCreateResponse:
    try:
        opts = JobOptions.model_validate(json.loads(options) if options.strip() else {})
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid options JSON: {e}") from e
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e

    raw_name = file.filename or "upload.bin"
    suffix = Path(raw_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    jdir = _job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=False)
    in_path = jdir / f"input{suffix}"
    out_path = jdir / "output.mp4"

    total = 0
    try:
        with open(in_path, "wb") as out_f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    shutil.rmtree(jdir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
                    )
                out_f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(jdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}") from e

    with _jobs_lock:
        JOBS[job_id] = {
            "status": "pending",
            "error": None,
            "output_path": None,
            "work_dir": str(jdir),
        }

    kwargs = opts.to_render_kwargs()
    background_tasks.add_task(_run_job_task, job_id, in_path, out_path, kwargs)
    return JobCreateResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    with _jobs_lock:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    dl = f"/api/jobs/{job_id}/download" if job["status"] == "done" else None
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
        download_path=dl,
    )


def _cleanup_job_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str, background_tasks: BackgroundTasks) -> FileResponse:
    with _jobs_lock:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not ready (status={job['status']!r})")
    out = job.get("output_path")
    if not out or not Path(out).is_file():
        raise HTTPException(status_code=500, detail="Output file missing")

    work_dir = job.get("work_dir")
    if work_dir:
        background_tasks.add_task(_cleanup_job_dir, work_dir)
    with _jobs_lock:
        JOBS.pop(job_id, None)

    return FileResponse(
        out,
        media_type="video/mp4",
        filename="beat_synced_squares_output.mp4",
    )


@app.get("/")
async def serve_ui() -> FileResponse:
    index = _ROOT / "web" / "index.html"
    if not index.is_file():
        return JSONResponse(
            status_code=503,
            content={"detail": "web/index.html missing — add the frontend bundle."},
        )
    return FileResponse(index)
