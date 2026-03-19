import json
import csv
import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.db.database import SessionLocal
from app.db.crud import (
    get_invoice_record_by_id,
    get_all_invoice_records,
    get_invoice_records_by_status,
    get_processing_run_by_id,
    get_review_actions_by_run_id,
)
from app.services.export_service import build_canonical_invoice_payload

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/invoice/{record_id}")
def export_single_invoice(record_id: int):
    db = SessionLocal()
    try:
        record = get_invoice_record_by_id(db, record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        run = get_processing_run_by_id(db, record.run_id) if record.run_id else None
        actions = get_review_actions_by_run_id(db, record.run_id) if record.run_id else []

        payload = build_canonical_invoice_payload(record, run, actions)

        return Response(
            content=json.dumps(payload, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="invoice_{record_id}_canonical.json"'}
        )
    finally:
        db.close()


@router.get("/invoices")
def export_multiple_invoices(status: str = "approved"):
    db = SessionLocal()
    try:
        if status == "all":
            records = get_all_invoice_records(db)
        else:
            records = get_invoice_records_by_status(db, status)

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "record_id",
            "document_id",
            "vendor_name",
            "invoice_number",
            "invoice_date",
            "total",
            "currency",
            "status",
            "confidence_score",
        ])

        for record in records:
            invoice_data = json.loads(record.invoice_data_json) if record.invoice_data_json else {}
            validation = json.loads(record.validation_result_json) if record.validation_result_json else {}

            normalized = invoice_data.get("normalized_invoice_fields", {})

            writer.writerow([
                record.id,
                record.document_id,
                normalized.get("vendor_name"),
                normalized.get("invoice_number"),
                normalized.get("invoice_date"),
                normalized.get("total"),
                normalized.get("currency"),
                record.status,
                validation.get("confidence_score"),
            ])

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="invoices_{status}.csv"'}
        )
    finally:
        db.close()