import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Document, ProcessingRun, ProcessingStep
from app.db.crud import create_invoice_record
from app.schemas.invoice import InvoiceData
from app.services.ai_extraction_service import process_document_with_ai
from app.services.pdf_service import extract_text_from_pdf, render_first_page_to_image
from app.services.validation_service import validate_invoice_data


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _start_step(
    db: Session,
    run_id: int,
    step_name: str,
    input_data: dict | None = None,
) -> ProcessingStep:
    step = ProcessingStep(
        run_id=run_id,
        step_name=step_name,
        status="processing",
        input_snapshot_json=_json_dumps(input_data or {}),
        started_at=datetime.utcnow(),
    )
    db.add(step)
    db.commit()
    db.refresh(step)
    return step


def _finish_step(
    db: Session,
    step: ProcessingStep,
    status: str,
    output_data: dict | None = None,
    error_message: str | None = None,
) -> ProcessingStep:
    step.status = status
    step.output_snapshot_json = _json_dumps(output_data or {})
    step.error_message = error_message
    step.finished_at = datetime.utcnow()
    db.commit()
    db.refresh(step)
    return step


def _create_document(
    db: Session,
    original_filename: str,
    saved_path: str,
    mime_type: str | None,
) -> Document:
    document = Document(
        original_filename=original_filename,
        file_path=saved_path,
        mime_type=mime_type,
        created_at=datetime.utcnow(),
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def _create_run(
    db: Session,
    document_id: int,
    model_name: str = "openai",
    pipeline_version: str = "v1",
) -> ProcessingRun:
    run = ProcessingRun(
        document_id=document_id,
        model_name=model_name,
        pipeline_version=pipeline_version,
        status="uploaded",
        started_at=datetime.utcnow(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


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
    compound_count = len([flag for flag in flags_present if flag in compound_high_risk_flags])

    if compound_count >= 3:
        adjusted_score = min(adjusted_score, 60.0)
    elif compound_count == 2:
        adjusted_score = min(adjusted_score, 66.0)

    return max(0.0, min(100.0, round(adjusted_score, 2)))


def _compute_review_status(
    classification: dict,
    normalized_invoice_data: dict,
    debug_info: dict,
) -> tuple[dict, str]:
    if not classification.get("is_invoice_like", False):
        validation_result = {
            "is_valid": False,
            "needs_review": False,
            "warnings": ["The uploaded document is not an invoice."],
            "confidence_score": 0,
            "review_reasons": ["Document classified as not invoice"],
            "missing_fields": [],
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

        if (
            no_items
            and no_date
            and used_fallback
            and used_vision
            and text_length < 1000
        ):
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
        flag for flag in field_review_flags
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


def process_uploaded_invoice(
    db: Session,
    original_filename: str,
    saved_path: str,
    mime_type: str | None = None,
) -> dict:
    document = _create_document(
        db=db,
        original_filename=original_filename,
        saved_path=saved_path,
        mime_type=mime_type,
    )
    run = _create_run(db=db, document_id=document.id)

    extracted_text = ""
    rendered_image_path = None
    classification: dict = {}
    invoice_data_dict: dict = {}
    normalized_invoice_data: dict = {}
    validation_result: dict = {}
    debug_info: dict = {}
    status = "processing"

    try:
        step = _start_step(
            db=db,
            run_id=run.id,
            step_name="text_extraction",
            input_data={
                "saved_path": saved_path,
                "mime_type": mime_type,
                "original_filename": original_filename,
            },
        )

        if original_filename.lower().endswith(".pdf"):
            rendered_image_path = str(Path(saved_path).with_suffix(".png"))
            render_first_page_to_image(saved_path, rendered_image_path)
            extracted_text = extract_text_from_pdf(saved_path)
        else:
            extracted_text = ""

        _finish_step(
            db=db,
            step=step,
            status="success",
            output_data={
                "text_length": len(extracted_text or ""),
                "preview": (extracted_text or "")[:1000],
                "rendered_image_path": rendered_image_path,
            },
        )

        run.status = "text_extracted"
        db.commit()

        step = _start_step(
            db=db,
            run_id=run.id,
            step_name="ai_processing",
            input_data={
                "text_length": len(extracted_text or ""),
                "pdf_path": saved_path,
                "rendered_image_path": rendered_image_path,
            },
        )

        ai_result = process_document_with_ai(extracted_text, pdf_path=saved_path)
        classification = ai_result.get("document_classification", {}) or {}
        invoice_data = ai_result.get("invoice_data")
        normalized_invoice_data = ai_result.get("normalized_invoice_data", {}) or {}
        debug_info = ai_result.get("debug_info", {}) or {}

        if hasattr(invoice_data, "model_dump"):
            invoice_data_dict = invoice_data.model_dump()
        elif isinstance(invoice_data, dict):
            invoice_data_dict = invoice_data
        else:
            invoice_data_dict = {}

        _finish_step(
            db=db,
            step=step,
            status="success",
            output_data={
                "classification": classification,
                "invoice_data": invoice_data_dict,
                "normalized_invoice_data": normalized_invoice_data,
                "debug_info": debug_info,
                "rendered_image_path": rendered_image_path,
            },
        )

        run.status = "extracted"
        run.used_fallback = bool(debug_info.get("used_fallback", False))
        run.fallback_reasons_json = _json_dumps(debug_info.get("fallback_reasons", []))
        run.classification_json = _json_dumps(classification)
        db.commit()

        step = _start_step(
            db=db,
            run_id=run.id,
            step_name="validation",
            input_data={
                "classification": classification,
                "normalized_invoice_data": normalized_invoice_data,
                "field_review_flags": debug_info.get("field_review_flags", []),
            },
        )

        validation_result, status = _compute_review_status(
            classification=classification,
            normalized_invoice_data=normalized_invoice_data,
            debug_info=debug_info,
        )

        _finish_step(
            db=db,
            step=step,
            status="success",
            output_data={
                "validation_result": validation_result,
                "final_status": status,
            },
        )

        run.status = "validated"
        run.validation_json = _json_dumps(validation_result)
        run.final_output_json = _json_dumps(
            {
                "document_classification": classification,
                "raw_invoice_fields": invoice_data_dict,
                "normalized_invoice_fields": normalized_invoice_data,
                "debug_info": debug_info,
                "status": status,
                "rendered_image_path": rendered_image_path,
            }
        )
        run.finished_at = datetime.utcnow()
        db.commit()

        legacy_record = create_invoice_record(
            db=db,
            document_id=document.id,
            run_id=run.id,
            original_filename=original_filename,
            saved_path=saved_path,
            extracted_text=extracted_text,
            invoice_data={
                "document_classification": classification,
                "raw_invoice_fields": invoice_data_dict,
                "normalized_invoice_fields": normalized_invoice_data,
                "debug_info": debug_info,
                "rendered_image_path": rendered_image_path,
            },
            validation_result=validation_result,
            status=status,
        )

        return {
            "message": "File uploaded, processed, and saved successfully",
            "document_id": document.id,
            "run_id": run.id,
            "record_id": legacy_record.id,
            "original_filename": original_filename,
            "saved_path": saved_path,
            "rendered_image_path": rendered_image_path,
            "document_classification": classification,
            "extracted_text_preview": (extracted_text or "")[:1000],
            "invoice_data": invoice_data_dict,
            "normalized_invoice_data": normalized_invoice_data,
            "validation_result": validation_result,
            "status": status,
            "debug_info": debug_info,
        }

    except Exception as e:
        failed_step = ProcessingStep(
            run_id=run.id,
            step_name="pipeline_error",
            status="failed",
            input_snapshot_json=_json_dumps(
                {
                    "original_filename": original_filename,
                    "saved_path": saved_path,
                    "rendered_image_path": rendered_image_path,
                }
            ),
            output_snapshot_json=_json_dumps({}),
            error_message=str(e),
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
        )
        db.add(failed_step)

        run.status = "failed"
        run.finished_at = datetime.utcnow()
        db.commit()
        raise