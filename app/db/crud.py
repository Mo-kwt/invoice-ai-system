import json

from sqlalchemy.orm import Session

from app.db.models import InvoiceRecord, ProcessingRun, ProcessingStep, ReviewAction


def create_invoice_record(
    db: Session,
    original_filename: str,
    saved_path: str,
    extracted_text: str,
    invoice_data: dict,
    validation_result: dict,
    status: str = "processed",
    document_id: int | None = None,
    run_id: int | None = None,
) -> InvoiceRecord:
    used_fallback = (
        invoice_data.get("debug_info", {}).get("used_fallback")
        if isinstance(invoice_data, dict)
        else None
    )

    print("CREATE_RECORD PARAM used_fallback:", used_fallback)

    record = InvoiceRecord(
        document_id=document_id,
        run_id=run_id,
        original_filename=original_filename,
        saved_path=saved_path,
        extracted_text=extracted_text,
        invoice_data_json=json.dumps(invoice_data, ensure_ascii=False),
        validation_result_json=json.dumps(validation_result, ensure_ascii=False),
        status=status,
        used_fallback=used_fallback,
    )

    print("BEFORE COMMIT used_fallback:", getattr(record, "used_fallback", "FIELD_NOT_PRESENT"))

    db.add(record)
    db.commit()
    db.refresh(record)

    print("AFTER SAVE used_fallback:", record.used_fallback)

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


def get_processing_run_by_id(db: Session, run_id: int):
    return db.query(ProcessingRun).filter(ProcessingRun.id == run_id).first()


def get_processing_steps_by_run_id(db: Session, run_id: int):
    return (
        db.query(ProcessingStep)
        .filter(ProcessingStep.run_id == run_id)
        .order_by(ProcessingStep.id.asc())
        .all()
    )


def create_review_action(
    db: Session,
    document_id: int | None,
    run_id: int | None,
    action_type: str,
    before_data: dict | None = None,
    after_data: dict | None = None,
    actor: str = "system",
    note: str | None = None,
):
    action = ReviewAction(
        document_id=document_id,
        run_id=run_id,
        action_type=action_type,
        actor=actor,
        before_json=json.dumps(before_data or {}, ensure_ascii=False),
        after_json=json.dumps(after_data or {}, ensure_ascii=False),
        note=note,
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def get_review_actions_by_run_id(db: Session, run_id: int):
    return (
        db.query(ReviewAction)
        .filter(ReviewAction.run_id == run_id)
        .order_by(ReviewAction.id.desc())
        .all()
    )