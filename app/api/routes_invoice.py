import json
from fastapi import APIRouter, HTTPException
from app.db.database import SessionLocal
from app.db.crud import (
    get_all_invoice_records,
    get_invoice_record_by_id,
    update_invoice_status,
)

router = APIRouter()


@router.get("/invoices")
def list_invoices():
    db = SessionLocal()
    try:
        records = get_all_invoice_records(db)

        return [
            {
                "id": record.id,
                "original_filename": record.original_filename,
                "saved_path": record.saved_path,
                "status": record.status,
                "created_at": record.created_at.isoformat() if record.created_at else None
            }
            for record in records
        ]
    finally:
        db.close()


@router.get("/invoices/{record_id}")
def get_invoice(record_id: int):
    db = SessionLocal()
    try:
        record = get_invoice_record_by_id(db, record_id)

        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        return {
            "id": record.id,
            "original_filename": record.original_filename,
            "saved_path": record.saved_path,
            "status": record.status,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "extracted_text": record.extracted_text,
            "invoice_data": json.loads(record.invoice_data_json) if record.invoice_data_json else {},
            "validation_result": json.loads(record.validation_result_json) if record.validation_result_json else {}
        }
    finally:
        db.close()


@router.post("/invoices/{record_id}/approve")
def approve_invoice(record_id: int):
    db = SessionLocal()
    try:
        record = update_invoice_status(db, record_id, "approved")
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        return {
            "message": "Invoice approved successfully",
            "id": record.id,
            "status": record.status
        }
    finally:
        db.close()


@router.post("/invoices/{record_id}/reject")
def reject_invoice(record_id: int):
    db = SessionLocal()
    try:
        record = update_invoice_status(db, record_id, "rejected")
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        return {
            "message": "Invoice rejected successfully",
            "id": record.id,
            "status": record.status
        }
    finally:
        db.close()