import json
import csv
import io

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
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
from app.services.ai_extraction_service import process_document_with_ai
from app.services.validation_service import validate_invoice_data
from app.schemas.invoice import InvoiceData

router = APIRouter()
templates = Jinja2Templates(directory="templates")


def _clean_optional_str(value: str | None):
    if value is None:
        return None
    value = value.strip()
    return value if value != "" else None


def _clean_optional_float(value: str | None):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_home(request: Request, status: str = "all"):
    db = SessionLocal()
    try:
        allowed_statuses = {"all", "approved", "rejected", "needs_review", "valid", "not_invoice"}
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

        ai_result = process_document_with_ai(extracted_text, pdf_path=saved_path)
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

        create_invoice_record(
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

        normalized_data = invoice_data_dict.get("normalized_invoice_fields", {})
        raw_data = invoice_data_dict.get("raw_invoice_fields", {})
        classification = invoice_data_dict.get("document_classification", {})

        if not normalized_data and not raw_data and isinstance(invoice_data_dict, dict):
            normalized_data = invoice_data_dict

        return templates.TemplateResponse(
            "invoice_detail.html",
            {
                "request": request,
                "record": record,
                "data": normalized_data,
                "raw_data": raw_data,
                "classification": classification,
                "validation_result": validation_result_dict,
            }
        )
    finally:
        db.close()


@router.post("/dashboard/invoices/{record_id}/update")
def dashboard_update_invoice(
    record_id: int,
    vendor_name: str = Form(""),
    invoice_number: str = Form(""),
    invoice_date: str = Form(""),
    subtotal: str = Form(""),
    tax: str = Form(""),
    discount: str = Form(""),
    total: str = Form(""),
    currency: str = Form(""),
):
    db = SessionLocal()
    try:
        record = get_invoice_record_by_id(db, record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        invoice_data_dict = json.loads(record.invoice_data_json) if record.invoice_data_json else {}
        validation_result_dict = json.loads(record.validation_result_json) if record.validation_result_json else {}

        normalized_data = invoice_data_dict.get("normalized_invoice_fields", {})
        raw_data = invoice_data_dict.get("raw_invoice_fields", {})
        classification = invoice_data_dict.get("document_classification", {})

        if not normalized_data:
            normalized_data = {}

        normalized_data["vendor_name"] = _clean_optional_str(vendor_name)
        normalized_data["invoice_number"] = _clean_optional_str(invoice_number)
        normalized_data["invoice_date"] = _clean_optional_str(invoice_date)
        normalized_data["subtotal"] = _clean_optional_float(subtotal)
        normalized_data["tax"] = _clean_optional_float(tax)
        normalized_data["discount"] = _clean_optional_float(discount)
        normalized_data["total"] = _clean_optional_float(total)
        normalized_data["currency"] = _clean_optional_str(currency)
        normalized_data["items"] = normalized_data.get("items", [])

        invoice_obj = InvoiceData(**normalized_data)
        validation_obj = validate_invoice_data(invoice_obj)
        validation_result_dict = validation_obj.model_dump()

        if classification.get("is_invoice_like", False):
            status = "needs_review" if validation_obj.needs_review else "valid"
        else:
            status = "not_invoice"

        updated_invoice_payload = {
            "document_classification": classification,
            "raw_invoice_fields": raw_data,
            "normalized_invoice_fields": normalized_data,
        }

        record.invoice_data_json = json.dumps(updated_invoice_payload, ensure_ascii=False)
        record.validation_result_json = json.dumps(validation_result_dict, ensure_ascii=False)
        record.status = status

        db.commit()

        return RedirectResponse(url=f"/dashboard/invoices/{record_id}", status_code=303)
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
        allowed_statuses = {"all", "approved", "rejected", "needs_review", "valid", "not_invoice"}
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
            "document_type",
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

            normalized = invoice_data.get("normalized_invoice_fields", {})
            classification = invoice_data.get("document_classification", {})

            if not normalized and isinstance(invoice_data, dict):
                normalized = invoice_data

            writer.writerow([
                record.id,
                record.original_filename,
                record.saved_path,
                record.status,
                record.created_at.isoformat() if record.created_at else "",
                classification.get("document_type"),
                normalized.get("vendor_name"),
                normalized.get("invoice_number"),
                normalized.get("invoice_date"),
                normalized.get("subtotal"),
                normalized.get("tax"),
                normalized.get("discount"),
                normalized.get("total"),
                normalized.get("currency"),
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