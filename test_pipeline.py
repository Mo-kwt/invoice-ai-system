from pprint import pprint
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.services.pipeline_service import process_uploaded_invoice

db: Session = SessionLocal()

result = process_uploaded_invoice(
    db=db,
    original_filename="INVOICE11.pdf",
    saved_path="storage/uploads/INVOICE11.pdf",
    mime_type="application/pdf",
)

print("\nSTATUS:")
print(result["status"])

print("\nVALIDATION RESULT:")
pprint(result["validation_result"])

print("\nFIELD REVIEW FLAGS:")
pprint(result["debug_info"].get("field_review_flags", []))

print("\nFIELD EVIDENCE:")
pprint(result["debug_info"].get("field_evidence", {}))

print("\nNORMALIZED INVOICE DATA:")
pprint(result["normalized_invoice_data"])