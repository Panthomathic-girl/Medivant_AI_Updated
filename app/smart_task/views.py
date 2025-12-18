# app/smart_task_prioritization/views.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid
from app.smart_task.agent import suggest_next_tasks
from app.smart_task.models import SmartTaskResponse

router = APIRouter(prefix="/smart-task", tags=["Smart Task Prioritization"])

UPLOAD_DIR = Path("uploads/smart_tasks")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Audio extensions & MIME types
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".webm"}
AUDIO_MIME = {
    "audio/mp3", "audio/mpeg", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/flac", "audio/aac", "audio/ogg", "audio/webm", "audio/m4a"
}

# Video extensions & MIME types
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}
VIDEO_MIME = {
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska",
    "video/webm", "video/mpeg", "video/x-m4v"
}

MAX_SIZE_MB = 150
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024


@router.post("/", response_model=SmartTaskResponse)
async def suggest_tasks_endpoint(file_upload: UploadFile = File(...)):
    filename = file_upload.filename or "recording.unknown"
    ext = Path(filename).suffix.lower()
    content_type = file_upload.content_type or "application/octet-stream"

    # Validate extension
    if ext not in AUDIO_EXT.union(VIDEO_EXT):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Read content
    content = await file_upload.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_SIZE_MB}MB)")

    file_id = uuid.uuid4()
    temp_path = UPLOAD_DIR / f"{file_id}{ext}"

    try:
        temp_path.write_bytes(content)

        result = suggest_next_tasks(
            file_path=str(temp_path),
            mime_type=content_type,
            analyze_func=None  # Let agent.py auto-pick
        )
        return SmartTaskResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass