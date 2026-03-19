from fastapi import APIRouter, File, HTTPException, UploadFile

from app.db.database import SessionLocal
from app.services.file_service import save_upload_file
from app.services.pipeline_service import process_uploaded_invoice

router = APIRouter()


@router.post("/upload-invoice")
def upload_invoice(file: UploadFile = File(...)):
    db = SessionLocal()

    try:
        saved_path = save_upload_file(file)

        result = process_uploaded_invoice(
            db=db,
            original_filename=file.filename,
            saved_path=saved_path,
            mime_type=file.content_type,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    finally:
        db.close()