import json
from sqlalchemy.orm import Session
from app.db.models import InvoiceRecord


def create_invoice_record(
    db: Session,
    original_filename: str,
    saved_path: str,
    extracted_text: str,
    invoice_data: dict,
    validation_result: dict,
    status: str = "processed"
) -> InvoiceRecord:
    record = InvoiceRecord(
        original_filename=original_filename,
        saved_path=saved_path,
        extracted_text=extracted_text,
        invoice_data_json=json.dumps(invoice_data, ensure_ascii=False),
        validation_result_json=json.dumps(validation_result, ensure_ascii=False),
        status=status
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_all_invoice_records(db: Session):
    return db.query(InvoiceRecord).order_by(InvoiceRecord.id.desc()).all()


def get_invoice_record_by_id(db: Session, record_id: int):
    return db.query(InvoiceRecord).filter(InvoiceRecord.id == record_id).first()

def update_invoice_status(db: Session, record_id: int, new_status: str):
    record = db.query(InvoiceRecord).filter(InvoiceRecord.id == record_id).first()
    if not record:
        return None

    record.status = new_status
    db.commit()
    db.refresh(record)
    return record

def get_invoice_records_by_status(db: Session, status: str):
    return (
        db.query(InvoiceRecord)
        .filter(InvoiceRecord.status == status)
        .order_by(InvoiceRecord.id.desc())
        .all()
    )