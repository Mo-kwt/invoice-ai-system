import json
import csv
import io
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from app.db.database import SessionLocal
from app.db.crud import (
    get_all_invoice_records,
    get_invoice_record_by_id,
    update_invoice_status,
    create_invoice_record,
    get_invoice_records_by_status,
)
from app.services.file_service import save_upload_file
from app.services.pdf_service import extract_text_from_pdf
from app.services.ai_extraction_service import extract_invoice_data_from_text
from app.services.validation_service import validate_invoice_data

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_home(request: Request, status: str = "all"):
    db = SessionLocal()
    try:
        allowed_statuses = {"all", "approved", "rejected", "needs_review", "valid"}

        if status not in allowed_statuses:
            status = "all"

        if status == "all":
            records = get_all_invoice_records(db)
        else:
            records = get_invoice_records_by_status(db, status)

        invoices = [
            {
                "id": record.id,
                "original_filename": record.original_filename,
                "status": record.status,
                "created_at": record.created_at.isoformat() if record.created_at else None
            }
            for record in records
        ]

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "invoices": invoices,
                "current_status": status,
            }
        )
    finally:
        db.close()


@router.post("/dashboard/upload")
def dashboard_upload_invoice(file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        saved_path = save_upload_file(file)

        extracted_text = ""
        if file.filename.lower().endswith(".pdf"):
            extracted_text = extract_text_from_pdf(saved_path)

        invoice_data = extract_invoice_data_from_text(extracted_text)
        validation_result = validate_invoice_data(invoice_data)

        create_invoice_record(
            db=db,
            original_filename=file.filename,
            saved_path=saved_path,
            extracted_text=extracted_text,
            invoice_data=invoice_data.model_dump(),
            validation_result=validation_result.model_dump(),
            status="needs_review" if validation_result.needs_review else "valid"
        )

        return RedirectResponse(url="/dashboard", status_code=303)
    finally:
        db.close()


@router.get("/dashboard/invoices/{record_id}", response_class=HTMLResponse)
def dashboard_invoice_detail(request: Request, record_id: int):
    db = SessionLocal()
    try:
        record = get_invoice_record_by_id(db, record_id)

        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        invoice_data_dict = json.loads(record.invoice_data_json) if record.invoice_data_json else {}
        validation_result_dict = json.loads(record.validation_result_json) if record.validation_result_json else {}

        invoice = {
            "id": record.id,
            "original_filename": record.original_filename,
            "saved_path": record.saved_path,
            "status": record.status,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "extracted_text": record.extracted_text or "",
            "invoice_data_dict": invoice_data_dict,
            "validation_result_dict": validation_result_dict,
            "invoice_data": json.dumps(invoice_data_dict, indent=2, ensure_ascii=False),
            "validation_result": json.dumps(validation_result_dict, indent=2, ensure_ascii=False)
        }

        return templates.TemplateResponse(
            "invoice_detail.html",
            {"request": request, "invoice": invoice}
        )
    finally:
        db.close()


@router.post("/dashboard/invoices/{record_id}/approve")
def dashboard_approve_invoice(record_id: int):
    db = SessionLocal()
    try:
        record = update_invoice_status(db, record_id, "approved")
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        return RedirectResponse(url=f"/dashboard/invoices/{record_id}", status_code=303)
    finally:
        db.close()


@router.post("/dashboard/invoices/{record_id}/reject")
def dashboard_reject_invoice(record_id: int):
    db = SessionLocal()
    try:
        record = update_invoice_status(db, record_id, "rejected")
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        return RedirectResponse(url=f"/dashboard/invoices/{record_id}", status_code=303)
    finally:
        db.close()


@router.get("/dashboard/invoices/{record_id}/download-json")
def download_invoice_json(record_id: int):
    db = SessionLocal()
    try:
        record = get_invoice_record_by_id(db, record_id)

        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        payload = {
            "id": record.id,
            "original_filename": record.original_filename,
            "saved_path": record.saved_path,
            "status": record.status,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "extracted_text": record.extracted_text or "",
            "invoice_data": json.loads(record.invoice_data_json) if record.invoice_data_json else {},
            "validation_result": json.loads(record.validation_result_json) if record.validation_result_json else {}
        }

        json_content = json.dumps(payload, indent=2, ensure_ascii=False)

        return Response(
            content=json_content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="invoice_{record_id}.json"'
            }
        )
    finally:
        db.close()


@router.get("/dashboard/download-csv")
def download_all_invoices_csv(status: str = "all"):
    db = SessionLocal()
    try:
        allowed_statuses = {"all", "approved", "rejected", "needs_review", "valid"}

        if status not in allowed_statuses:
            status = "all"

        if status == "all":
            records = get_all_invoice_records(db)
        else:
            records = get_invoice_records_by_status(db, status)

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "id",
            "original_filename",
            "saved_path",
            "status",
            "created_at",
            "vendor_name",
            "invoice_number",
            "invoice_date",
            "subtotal",
            "tax",
            "discount",
            "total",
            "currency"
        ])

        for record in records:
            invoice_data = json.loads(record.invoice_data_json) if record.invoice_data_json else {}

            writer.writerow([
                record.id,
                record.original_filename,
                record.saved_path,
                record.status,
                record.created_at.isoformat() if record.created_at else "",
                invoice_data.get("vendor_name"),
                invoice_data.get("invoice_number"),
                invoice_data.get("invoice_date"),
                invoice_data.get("subtotal"),
                invoice_data.get("tax"),
                invoice_data.get("discount"),
                invoice_data.get("total"),
                invoice_data.get("currency"),
            ])

        csv_content = output.getvalue()
        output.close()

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="invoices_{status}.csv"'
            }
        )
    finally:
        db.close()