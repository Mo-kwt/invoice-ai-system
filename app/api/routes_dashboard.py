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
    get_invoice_records_by_status,
    get_processing_run_by_id,
    get_processing_steps_by_run_id,
    create_review_action,
    get_review_actions_by_run_id,
)
from app.services.file_service import save_upload_file
from app.services.pipeline_service import process_uploaded_invoice
from app.services.integration_service import append_invoice_to_excel
from app.services.export_service import build_canonical_invoice_payload
from app.schemas.invoice import InvoiceData
from app.services.validation_service import validate_invoice_data

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


def _safe_json_loads(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _extract_dashboard_row(record):
    invoice_data_dict = _safe_json_loads(record.invoice_data_json)
    validation_result_dict = _safe_json_loads(record.validation_result_json)

    normalized_data = invoice_data_dict.get("normalized_invoice_fields", {})
    classification = invoice_data_dict.get("document_classification", {})
    debug_info = invoice_data_dict.get("debug_info", {})

    if not normalized_data and isinstance(invoice_data_dict, dict):
        normalized_data = invoice_data_dict

    confidence_score = validation_result_dict.get("confidence_score", 0)
    needs_review = validation_result_dict.get("needs_review", False)
    used_fallback = debug_info.get("used_fallback", False)

    if confidence_score < 60:
        review_priority = "عالية"
    elif confidence_score < 85:
        review_priority = "متوسطة"
    else:
        review_priority = "منخفضة"

    return {
        "id": record.id,
        "original_filename": record.original_filename,
        "status": record.status,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "document_type": classification.get("document_type", "-"),
        "vendor_name": normalized_data.get("vendor_name"),
        "invoice_number": normalized_data.get("invoice_number"),
        "invoice_date": normalized_data.get("invoice_date"),
        "total": normalized_data.get("total"),
        "currency": normalized_data.get("currency"),
        "confidence_score": confidence_score,
        "needs_review": needs_review,
        "review_priority": review_priority,
        "used_fallback": used_fallback,
    }


def _build_record_snapshot(record):
    return {
        "record_id": record.id,
        "document_id": record.document_id,
        "run_id": record.run_id,
        "original_filename": record.original_filename,
        "saved_path": record.saved_path,
        "status": record.status,
        "invoice_data": _safe_json_loads(record.invoice_data_json),
        "validation_result": _safe_json_loads(record.validation_result_json),
    }


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

        invoices = [_extract_dashboard_row(record) for record in records]

        invoices.sort(
            key=lambda x: (
                0 if x["needs_review"] else 1,
                x["confidence_score"],
                -(x["id"] or 0),
            )
        )

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

        process_uploaded_invoice(
            db=db,
            original_filename=file.filename,
            saved_path=saved_path,
            mime_type=file.content_type,
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

        invoice_data_dict = _safe_json_loads(record.invoice_data_json)
        validation_result_dict = _safe_json_loads(record.validation_result_json)

        normalized_data = invoice_data_dict.get("normalized_invoice_fields", {})
        raw_data = invoice_data_dict.get("raw_invoice_fields", {})
        classification = invoice_data_dict.get("document_classification", {})
        debug_info = invoice_data_dict.get("debug_info", {})

        if not normalized_data and not raw_data and isinstance(invoice_data_dict, dict):
            normalized_data = invoice_data_dict

        processing_run = None
        processing_steps = []
        fallback_reasons = []
        review_actions = []

        if record.run_id:
            processing_run = get_processing_run_by_id(db, record.run_id)
            processing_steps = get_processing_steps_by_run_id(db, record.run_id)
            review_actions = get_review_actions_by_run_id(db, record.run_id)

            if processing_run and processing_run.fallback_reasons_json:
                try:
                    fallback_reasons = json.loads(processing_run.fallback_reasons_json)
                except Exception:
                    fallback_reasons = []

        step_rows = []
        for step in processing_steps:
            step_rows.append(
                {
                    "id": step.id,
                    "step_name": step.step_name,
                    "status": step.status,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "finished_at": step.finished_at.isoformat() if step.finished_at else None,
                    "error_message": step.error_message,
                    "input_snapshot": _safe_json_loads(step.input_snapshot_json),
                    "output_snapshot": _safe_json_loads(step.output_snapshot_json),
                }
            )

        run_view = None
        if processing_run:
            run_view = {
                "id": processing_run.id,
                "document_id": processing_run.document_id,
                "status": processing_run.status,
                "pipeline_version": processing_run.pipeline_version,
                "model_name": processing_run.model_name,
                "used_fallback": processing_run.used_fallback,
                "fallback_reasons": fallback_reasons,
                "started_at": processing_run.started_at.isoformat() if processing_run.started_at else None,
                "finished_at": processing_run.finished_at.isoformat() if processing_run.finished_at else None,
            }

        review_rows = []
        for action in review_actions:
            review_rows.append(
                {
                    "id": action.id,
                    "action_type": action.action_type,
                    "actor": action.actor,
                    "note": action.note,
                    "created_at": action.created_at.isoformat() if action.created_at else None,
                    "before_data": _safe_json_loads(action.before_json),
                    "after_data": _safe_json_loads(action.after_json),
                }
            )

        return templates.TemplateResponse(
            "invoice_detail.html",
            {
                "request": request,
                "record": record,
                "data": normalized_data,
                "raw_data": raw_data,
                "classification": classification,
                "validation_result": validation_result_dict,
                "debug_info": debug_info,
                "run_view": run_view,
                "step_rows": step_rows,
                "review_rows": review_rows,
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

        before_snapshot = _build_record_snapshot(record)

        invoice_data_dict = _safe_json_loads(record.invoice_data_json)
        raw_data = invoice_data_dict.get("raw_invoice_fields", {})
        classification = invoice_data_dict.get("document_classification", {})
        debug_info = invoice_data_dict.get("debug_info", {})
        normalized_data = invoice_data_dict.get("normalized_invoice_fields", {})

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
            "debug_info": debug_info,
        }

        record.invoice_data_json = json.dumps(updated_invoice_payload, ensure_ascii=False)
        record.validation_result_json = json.dumps(validation_result_dict, ensure_ascii=False)
        record.status = status

        db.commit()
        db.refresh(record)

        after_snapshot = _build_record_snapshot(record)

        create_review_action(
            db=db,
            document_id=record.document_id,
            run_id=record.run_id,
            action_type="edited",
            before_data=before_snapshot,
            after_data=after_snapshot,
            actor="reviewer",
            note="Manual field update from dashboard",
        )

        return RedirectResponse(url=f"/dashboard/invoices/{record_id}", status_code=303)
    finally:
        db.close()


@router.post("/dashboard/invoices/{record_id}/approve")
def dashboard_approve_invoice(record_id: int):
    db = SessionLocal()
    try:
        record = get_invoice_record_by_id(db, record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        before_snapshot = _build_record_snapshot(record)

        record = update_invoice_status(db, record_id, "approved")
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        db.refresh(record)
        after_snapshot = _build_record_snapshot(record)

        create_review_action(
            db=db,
            document_id=record.document_id,
            run_id=record.run_id,
            action_type="approved",
            before_data=before_snapshot,
            after_data=after_snapshot,
            actor="approver",
            note="Invoice approved from dashboard",
        )

        run = None
        actions = []

        if record.run_id:
            run = get_processing_run_by_id(db, record.run_id)
            actions = get_review_actions_by_run_id(db, record.run_id)

        payload = build_canonical_invoice_payload(record, run, actions)
        append_invoice_to_excel(payload)

        return RedirectResponse(url=f"/dashboard/invoices/{record_id}", status_code=303)
    finally:
        db.close()


@router.post("/dashboard/invoices/{record_id}/reject")
def dashboard_reject_invoice(record_id: int):
    db = SessionLocal()
    try:
        record = get_invoice_record_by_id(db, record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        before_snapshot = _build_record_snapshot(record)

        record = update_invoice_status(db, record_id, "rejected")
        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        after_snapshot = _build_record_snapshot(record)

        create_review_action(
            db=db,
            document_id=record.document_id,
            run_id=record.run_id,
            action_type="rejected",
            before_data=before_snapshot,
            after_data=after_snapshot,
            actor="approver",
            note="Invoice rejected from dashboard",
        )

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
            "document_id": record.document_id,
            "run_id": record.run_id,
            "original_filename": record.original_filename,
            "saved_path": record.saved_path,
            "status": record.status,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "extracted_text": record.extracted_text or "",
            "invoice_data": _safe_json_loads(record.invoice_data_json),
            "validation_result": _safe_json_loads(record.validation_result_json),
        }

        json_content = json.dumps(payload, indent=2, ensure_ascii=False)

        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="invoice_{record_id}.json"'}
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
            "document_id",
            "run_id",
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
            "currency",
            "confidence_score",
            "needs_review",
            "used_fallback",
        ])

        for record in records:
            invoice_data = _safe_json_loads(record.invoice_data_json)
            validation_result = _safe_json_loads(record.validation_result_json)

            normalized = invoice_data.get("normalized_invoice_fields", {})
            classification = invoice_data.get("document_classification", {})
            debug_info = invoice_data.get("debug_info", {})

            if not normalized and isinstance(invoice_data, dict):
                normalized = invoice_data

            writer.writerow([
                record.id,
                record.document_id,
                record.run_id,
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
                validation_result.get("confidence_score", 0),
                validation_result.get("needs_review", False),
                debug_info.get("used_fallback", False),
            ])

        csv_content = output.getvalue()
        output.close()

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="invoices_{status}.csv"'}
        )
    finally:
        db.close()