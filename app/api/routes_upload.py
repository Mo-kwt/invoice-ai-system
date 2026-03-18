from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file_service import save_upload_file
from app.services.pdf_service import extract_text_from_pdf
from app.services.ai_extraction_service import extract_invoice_data_from_text
from app.services.validation_service import validate_invoice_data
from app.db.database import SessionLocal
from app.db.crud import create_invoice_record

router = APIRouter()


@router.post("/upload-invoice")
def upload_invoice(file: UploadFile = File(...)):
    db = SessionLocal()

    try:
        saved_path = save_upload_file(file)

        extracted_text = ""
        if file.filename.lower().endswith(".pdf"):
            extracted_text = extract_text_from_pdf(saved_path)

        invoice_data = extract_invoice_data_from_text(extracted_text)
        validation_result = validate_invoice_data(invoice_data)

        record = create_invoice_record(
            db=db,
            original_filename=file.filename,
            saved_path=saved_path,
            extracted_text=extracted_text,
            invoice_data=invoice_data.model_dump(),
            validation_result=validation_result.model_dump(),
            status="needs_review" if validation_result.needs_review else "valid"
        )

        return {
            "message": "File uploaded, processed, and saved successfully",
            "record_id": record.id,
            "original_filename": file.filename,
            "saved_path": saved_path,
            "extracted_text_preview": extracted_text[:1000],
            "invoice_data": invoice_data.model_dump(),
            "validation_result": validation_result.model_dump()
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        db.close()