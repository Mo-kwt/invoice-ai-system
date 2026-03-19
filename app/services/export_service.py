import json


def build_canonical_invoice_payload(record, processing_run=None, review_actions=None):
    invoice_data = json.loads(record.invoice_data_json) if record.invoice_data_json else {}
    validation_result = json.loads(record.validation_result_json) if record.validation_result_json else {}

    normalized = invoice_data.get("normalized_invoice_fields", {})
    classification = invoice_data.get("document_classification", {})
    debug_info = invoice_data.get("debug_info", {})

    fallback_reasons = []
    if processing_run and processing_run.fallback_reasons_json:
        try:
            fallback_reasons = json.loads(processing_run.fallback_reasons_json)
        except:
            fallback_reasons = []

    audit_summary = []
    if review_actions:
        for a in review_actions:
            audit_summary.append({
                "action": a.action_type,
                "actor": a.actor,
                "timestamp": a.created_at.isoformat() if a.created_at else None
            })

    return {
        "document": {
            "record_id": record.id,
            "document_id": record.document_id,
            "filename": record.original_filename,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        },
        "processing": {
            "run_id": record.run_id,
            "status": processing_run.status if processing_run else None,
            "pipeline_version": processing_run.pipeline_version if processing_run else None,
            "model": processing_run.model_name if processing_run else None,
            "used_fallback": processing_run.used_fallback if processing_run else False,
            "fallback_reasons": fallback_reasons,
        },
        "classification": classification,
        "extraction": normalized,
        "validation": validation_result,
        "workflow": {
            "status": record.status,
            "needs_review": validation_result.get("needs_review"),
            "confidence_score": validation_result.get("confidence_score"),
        },
        "audit": audit_summary,
        "debug": debug_info
    }