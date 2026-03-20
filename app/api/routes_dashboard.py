import csv
import io
import json
import shutil
from pathlib import Path

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
from app.services.pdf_service import extract_text_from_pdf, render_first_page_to_image
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


def _safe_json_loads(value: str | None):
    if not value:
        return {}
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, (dict, list)) else {}
    except Exception:
        return {}


def _deduplicate_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _apply_field_review_flag_adjustments(
    confidence_score: float,
    field_review_flags: list[str],
) -> float:
    adjusted_score = float(confidence_score)

    flag_penalties = {
        "invoice_number_missing": 18,
        "invoice_number_missing_but_candidate_exists": 16,
        "invoice_number_ambiguous": 18,
        "invoice_number_low_confidence": 12,
        "invoice_number_differs_from_top_candidate": 10,
        "invoice_date_missing": 18,
        "invoice_date_missing_but_candidate_exists": 16,
        "invoice_date_ambiguous": 18,
        "invoice_date_low_confidence": 10,
        "invoice_date_differs_from_top_candidate": 8,
        "items_missing_but_table_detected": 12,
        "items_not_reliably_detected": 10,
    }

    flags_present = [flag for flag in field_review_flags if flag in flag_penalties]

    for flag in flags_present:
        adjusted_score -= flag_penalties[flag]

    if flags_present:
        adjusted_score = min(adjusted_score, 79.0)

    if any(
        flag in {
            "invoice_number_ambiguous",
            "invoice_date_ambiguous",
            "invoice_number_missing_but_candidate_exists",
            "invoice_date_missing_but_candidate_exists",
            "invoice_number_low_confidence",
            "invoice_date_low_confidence",
        }
        for flag in flags_present
    ):
        adjusted_score = min(adjusted_score, 72.0)

    if any(
        flag in {"invoice_number_missing", "invoice_date_missing"}
        for flag in flags_present
    ):
        adjusted_score = min(adjusted_score, 68.0)

    compound_high_risk_flags = {
        "invoice_number_missing",
        "invoice_number_missing_but_candidate_exists",
        "invoice_number_ambiguous",
        "invoice_number_low_confidence",
        "invoice_date_missing",
        "invoice_date_missing_but_candidate_exists",
        "invoice_date_ambiguous",
        "invoice_date_low_confidence",
        "items_missing_but_table_detected",
        "items_not_reliably_detected",
    }

    compound_count = len(
        [flag for flag in flags_present if flag in compound_high_risk_flags]
    )

    if compound_count >= 3:
        adjusted_score = min(adjusted_score, 60.0)
    elif compound_count == 2:
        adjusted_score = min(adjusted_score, 66.0)

    return max(0.0, min(100.0, round(adjusted_score, 2)))


def _compute_dashboard_review_status(
    classification: dict,
    normalized_invoice_data: dict,
    debug_info: dict,
) -> tuple[dict, str]:
    if not classification.get("is_invoice_like", False):
        validation_result = {
            "is_valid": False,
            "needs_review": False,
            "warnings": ["The uploaded document is not an invoice."],
            "review_reasons": ["Document classified as not invoice"],
            "missing_fields": [],
            "confidence_score": 0,
        }
        return validation_result, "not_invoice"

    normalized_invoice_obj = InvoiceData(**normalized_invoice_data)
    validation_obj = validate_invoice_data(normalized_invoice_obj)
    validation_result = validation_obj.model_dump()

    field_review_flags = debug_info.get("field_review_flags", []) or []

    critical_flags = {
        "invoice_number_missing",
        "invoice_number_missing_but_candidate_exists",
        "invoice_number_ambiguous",
        "invoice_number_low_confidence",
        "invoice_date_missing",
        "invoice_date_missing_but_candidate_exists",
        "invoice_date_ambiguous",
        "invoice_date_low_confidence",
        "items_missing_but_table_detected",
        "items_not_reliably_detected",
    }

    needs_review_from_fields = any(flag in critical_flags for flag in field_review_flags)

    adjusted_confidence_score = _apply_field_review_flag_adjustments(
        confidence_score=validation_result.get("confidence_score", 0),
        field_review_flags=field_review_flags,
    )

    suspicious_total = False
    if normalized_invoice_data.get("total") is not None:
        no_items = not normalized_invoice_data.get("items")
        no_date = not normalized_invoice_data.get("invoice_date")
        used_fallback = debug_info.get("used_fallback", False)
        used_vision = debug_info.get("used_vision_fallback", False)
        text_length = debug_info.get("text_length", 0)

        if no_items and no_date and used_fallback and used_vision and text_length < 1000:
            suspicious_total = True

    if suspicious_total:
        adjusted_confidence_score = min(adjusted_confidence_score, 65.0)

    final_needs_review = (
        validation_result.get("needs_review", False)
        or needs_review_from_fields
        or suspicious_total
        or adjusted_confidence_score < 75
    )

    review_reasons = list(validation_result.get("review_reasons", []) or [])

    additional_field_review_flags = [
        flag
        for flag in field_review_flags
        if flag in critical_flags and flag not in review_reasons
    ]
    if additional_field_review_flags:
        review_reasons.extend(additional_field_review_flags)

    if suspicious_total and "total_low_confidence_due_to_weak_ocr" not in review_reasons:
        review_reasons.append("total_low_confidence_due_to_weak_ocr")

    validation_result["confidence_score"] = adjusted_confidence_score
    validation_result["needs_review"] = final_needs_review
    validation_result["review_reasons"] = _deduplicate_keep_order(review_reasons)

    status = "needs_review" if final_needs_review else "valid"
    return validation_result, status


def _build_rendered_preview(saved_path: str, original_filename: str) -> str | None:
    source_path = Path(saved_path)
    processed_dir = Path("storage/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    suffix = source_path.suffix.lower()
    preview_name = f"{source_path.stem}_preview.png"

    if suffix == ".pdf":
        output_path = processed_dir / preview_name
        return render_first_page_to_image(str(source_path), str(output_path))

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        output_path = processed_dir / f"{source_path.stem}_preview{suffix}"
        shutil.copyfile(source_path, output_path)
        return str(output_path)

    return None


def _build_dashboard_row(record):
    invoice_data_dict = _safe_json_loads(record.invoice_data_json)
    validation_result_dict = _safe_json_loads(record.validation_result_json)

    normalized_data = {}
    raw_data = {}
    classification = {}

    if isinstance(invoice_data_dict, dict):
        normalized_data = invoice_data_dict.get("normalized_invoice_fields", {}) or {}
        raw_data = invoice_data_dict.get("raw_invoice_fields", {}) or {}
        classification = invoice_data_dict.get("document_classification", {}) or {}

        if not normalized_data and not raw_data:
            normalized_data = invoice_data_dict

    confidence_score = None
    warnings_count = 0
    review_reasons_count = 0

    if isinstance(validation_result_dict, dict):
        confidence_score = validation_result_dict.get("confidence_score")
        warnings_count = len(validation_result_dict.get("warnings", []) or [])
        review_reasons_count = len(validation_result_dict.get("review_reasons", []) or [])

    source_data = normalized_data or raw_data or {}

    return {
        "id": record.id,
        "original_filename": record.original_filename,
        "saved_path": record.saved_path,
        "status": record.status,
        "created_at": record.created_at,
        "confidence_score": confidence_score if confidence_score is not None else 0,
        "document_type": classification.get("document_type"),
        "vendor_name": source_data.get("vendor_name"),
        "invoice_number": source_data.get("invoice_number"),
        "invoice_date": source_data.get("invoice_date"),
        "total": source_data.get("total"),
        "currency": source_data.get("currency"),
        "warnings_count": warnings_count,
        "review_reasons_count": review_reasons_count,
    }


def _build_detail_context(record):
    invoice_data_dict = _safe_json_loads(record.invoice_data_json)
    validation_result_dict = _safe_json_loads(record.validation_result_json)

    normalized_data = (
        invoice_data_dict.get("normalized_invoice_fields", {})
        if isinstance(invoice_data_dict, dict)
        else {}
    )
    raw_data = (
        invoice_data_dict.get("raw_invoice_fields", {})
        if isinstance(invoice_data_dict, dict)
        else {}
    )
    classification = (
        invoice_data_dict.get("document_classification", {})
        if isinstance(invoice_data_dict, dict)
        else {}
    )
    debug_info = (
        invoice_data_dict.get("debug_info", {})
        if isinstance(invoice_data_dict, dict)
        else {}
    )
    rendered_image_path = (
        invoice_data_dict.get("rendered_image_path")
        if isinstance(invoice_data_dict, dict)
        else None
    )

    if not normalized_data and not raw_data and isinstance(invoice_data_dict, dict):
        normalized_data = invoice_data_dict

    field_evidence = debug_info.get("field_evidence", {}) if isinstance(debug_info, dict) else {}
    field_review_flags = debug_info.get("field_review_flags", []) if isinstance(debug_info, dict) else []
    review_reasons = validation_result_dict.get("review_reasons", []) if isinstance(validation_result_dict, dict) else []
    warnings = validation_result_dict.get("warnings", []) if isinstance(validation_result_dict, dict) else []

    return {
        "record": record,
        "data": normalized_data or {},
        "raw_data": raw_data or {},
        "classification": classification or {},
        "validation_result": validation_result_dict or {},
        "debug_info": debug_info or {},
        "field_evidence": field_evidence or {},
        "field_review_flags": field_review_flags or [],
        "review_reasons": review_reasons or [],
        "warnings": warnings or [],
        "rendered_image_path": rendered_image_path,
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

        invoices = [_build_dashboard_row(record) for record in records]

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "invoices": invoices,
                "current_status": status,
            },
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
        debug_info = ai_result.get("debug_info", {})

        rendered_image_path = _build_rendered_preview(saved_path, file.filename)

        validation_result, status = _compute_dashboard_review_status(
            classification=classification,
            normalized_invoice_data=normalized_invoice_data,
            debug_info=debug_info,
        )

        create_invoice_record(
            db=db,
            original_filename=file.filename,
            saved_path=saved_path,
            extracted_text=extracted_text,
            invoice_data={
                "document_classification": classification,
                "raw_invoice_fields": invoice_data.model_dump() if hasattr(invoice_data, "model_dump") else invoice_data,
                "normalized_invoice_fields": normalized_invoice_data,
                "debug_info": debug_info,
                "rendered_image_path": rendered_image_path,
            },
            validation_result=validation_result,
            status=status,
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

        context = _build_detail_context(record)
        context["request"] = request
        return templates.TemplateResponse("invoice_detail.html", context)
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

        invoice_data_dict = _safe_json_loads(record.invoice_data_json)

        normalized_data = invoice_data_dict.get("normalized_invoice_fields", {}) if isinstance(invoice_data_dict, dict) else {}
        raw_data = invoice_data_dict.get("raw_invoice_fields", {}) if isinstance(invoice_data_dict, dict) else {}
        classification = invoice_data_dict.get("document_classification", {}) if isinstance(invoice_data_dict, dict) else {}
        debug_info = invoice_data_dict.get("debug_info", {}) if isinstance(invoice_data_dict, dict) else {}
        rendered_image_path = invoice_data_dict.get("rendered_image_path") if isinstance(invoice_data_dict, dict) else None

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

        validation_result_dict, status = _compute_dashboard_review_status(
            classification=classification,
            normalized_invoice_data=normalized_data,
            debug_info=debug_info,
        )

        updated_invoice_payload = {
            "document_classification": classification,
            "raw_invoice_fields": raw_data,
            "normalized_invoice_fields": normalized_data,
            "debug_info": debug_info,
            "rendered_image_path": rendered_image_path,
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
            "invoice_data": _safe_json_loads(record.invoice_data_json),
            "validation_result": _safe_json_loads(record.validation_result_json),
        }

        json_content = json.dumps(payload, indent=2, ensure_ascii=False)

        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="invoice_{record_id}.json"'},
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

        writer.writerow(
            [
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
                "currency",
                "confidence_score",
            ]
        )

        for record in records:
            invoice_data = _safe_json_loads(record.invoice_data_json)
            validation_result = _safe_json_loads(record.validation_result_json)

            normalized = invoice_data.get("normalized_invoice_fields", {}) if isinstance(invoice_data, dict) else {}
            classification = invoice_data.get("document_classification", {}) if isinstance(invoice_data, dict) else {}

            if not normalized and isinstance(invoice_data, dict):
                normalized = invoice_data

            writer.writerow(
                [
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
                    validation_result.get("confidence_score") if isinstance(validation_result, dict) else "",
                ]
            )

        csv_content = output.getvalue()
        output.close()

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="invoices_{status}.csv"'},
        )
    finally:
        db.close()