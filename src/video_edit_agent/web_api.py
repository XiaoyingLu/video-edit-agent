"""Small HTTP API for the local-first web editor.

The API manages project folders and source-media ingestion. Editing and analysis
remain delegated to the existing video-agent pipeline.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECTS_DIR = Path(os.environ.get("VIDEO_PROJECTS_DIR", "/workspace/projects")).resolve()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}

app = FastAPI(title="Video Edit Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    brief: str = Field(default="", max_length=5000)
    target_duration: str = Field(default="60–90 seconds", max_length=40)
    format: str = Field(default="16:9 · YouTube", max_length=40)


def _project_path(project_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    path = (PROJECTS_DIR / project_id).resolve()
    if not path.is_relative_to(PROJECTS_DIR):
        raise HTTPException(status_code=404, detail="Project not found")
    return path


def _read_project(path: Path) -> dict:
    metadata = path / "project.json"
    if not metadata.is_file():
        raise HTTPException(status_code=404, detail="Project not found")
    return json.loads(metadata.read_text(encoding="utf-8"))


def _write_project(path: Path, project: dict) -> None:
    temporary = path / "project.json.tmp"
    temporary.write_text(json.dumps(project, indent=2), encoding="utf-8")
    temporary.replace(path / "project.json")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/projects")
def list_projects() -> list[dict]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    projects = []
    for directory in PROJECTS_DIR.iterdir():
        if directory.is_dir() and (directory / "project.json").is_file():
            try:
                projects.append(_read_project(directory))
            except (OSError, json.JSONDecodeError):
                continue
    return sorted(projects, key=lambda item: item["updated_at"], reverse=True)


@app.post("/api/projects", status_code=201)
def create_project(request: ProjectCreate) -> dict:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    project_id = uuid.uuid4().hex
    path = PROJECTS_DIR / project_id
    (path / "source").mkdir(parents=True)
    now = datetime.now(UTC).isoformat()
    project = {
        "id": project_id,
        "name": request.name.strip(),
        "brief": request.brief,
        "target_duration": request.target_duration,
        "format": request.format,
        "media": [],
        "created_at": now,
        "updated_at": now,
    }
    _write_project(path, project)
    return project


@app.post("/api/projects/{project_id}/media", status_code=201)
async def upload_media(
    project_id: str,
    files: Annotated[list[UploadFile], File()],
) -> dict:
    path = _project_path(project_id)
    project = _read_project(path)
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one video file")

    prepared = []
    for upload in files:
        original_name = Path(upload.filename or "").name
        extension = Path(original_name).suffix.lower()
        if not original_name or extension not in VIDEO_EXTENSIONS:
            for pending in files:
                await pending.close()
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video file: {original_name or 'unnamed file'}",
            )
        prepared.append((upload, original_name, extension))

    saved = []
    for upload, original_name, extension in prepared:
        destination = path / "source" / original_name
        if destination.exists():
            stem = Path(original_name).stem
            destination = path / "source" / f"{stem}-{uuid.uuid4().hex[:8]}{extension}"
        size = 0
        try:
            with destination.open("xb") as output:
                while chunk := await upload.read(CHUNK_SIZE):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{original_name} exceeds the 20 GB upload limit",
                        )
                    output.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        saved.append({"name": destination.name, "size_bytes": size, "path": str(destination)})

    project["media"].extend(saved)
    project["updated_at"] = datetime.now(UTC).isoformat()
    _write_project(path, project)
    return project
