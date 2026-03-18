from pathlib import Path
from uuid import uuid4
from fastapi import UploadFile
from app.config import settings


ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


def ensure_storage_dirs() -> None:
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.export_dir).mkdir(parents=True, exist_ok=True)


def save_upload_file(file: UploadFile) -> str:
    ensure_storage_dirs()

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type")

    unique_name = f"{uuid4().hex}{suffix}"
    target_path = Path(settings.upload_dir) / unique_name

    with target_path.open("wb") as buffer:
        buffer.write(file.file.read())

    return str(target_path)