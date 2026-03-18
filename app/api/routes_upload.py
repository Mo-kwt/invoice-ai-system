from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.file_service import save_upload_file
from app.services.pdf_service import extract_text_from_pdf
from app.services.ai_extraction_service import process_document_with_ai
from app.services.validation_service import validate_invoice_data
from app.db.database import SessionLocal
from app.db.crud import create_invoice_record
from app.schemas.invoice import InvoiceData

router = APIRouter()


@router.post("/upload-invoice")
def upload_invoice(file: UploadFile = File(...)):
    db = SessionLocal()

    try:
        saved_path = save_upload_file(file)

        extracted_text = ""
        if file.filename.lower().endswith(".pdf"):
            extracted_text = extract_text_from_pdf(saved_path)

        ai_result = process_document_with_ai(extracted_text)
        classification = ai_result["document_classification"]
        invoice_data = ai_result["invoice_data"]
        normalized_invoice_data = ai_result["normalized_invoice_data"]

        if not classification.get("is_invoice_like", False):
            validation_result = {
                "is_valid": False,
                "needs_review": False,
                "warnings": ["The uploaded document is not an invoice."]
            }
            status = "not_invoice"
        else:
            normalized_invoice_obj = InvoiceData(**normalized_invoice_data)
            validation_obj = validate_invoice_data(normalized_invoice_obj)
            validation_result = validation_obj.model_dump()
            status = "needs_review" if validation_obj.needs_review else "valid"

        record = create_invoice_record(
            db=db,
            original_filename=file.filename,
            saved_path=saved_path,
            extracted_text=extracted_text,
            invoice_data={
                "document_classification": classification,
                "raw_invoice_fields": invoice_data.model_dump(),
                "normalized_invoice_fields": normalized_invoice_data,
            },
            validation_result=validation_result,
            status=status
        )

        return {
            "message": "File uploaded, processed, and saved successfully",
            "record_id": record.id,
            "original_filename": file.filename,
            "saved_path": saved_path,
            "document_classification": classification,
            "extracted_text_preview": extracted_text[:1000],
            "invoice_data": invoice_data.model_dump(),
            "normalized_invoice_data": normalized_invoice_data,
            "validation_result": validation_result,
            "status": status,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        db.close()