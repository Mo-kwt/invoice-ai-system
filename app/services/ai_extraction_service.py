import json
from openai import OpenAI

from app.config import settings
from app.schemas.invoice import InvoiceData
from app.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
)
from app.services.document_classifier_service import classify_document
from app.services.normalization_service import normalize_invoice_data
from app.services.postprocessing_service import enrich_normalized_invoice_data

client = OpenAI(api_key=settings.openai_api_key)


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_invoice_data_from_text(text: str) -> InvoiceData:
    if not text or not text.strip():
        return InvoiceData()

    prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
        document_text=text[:12000]
    )

    response = client.chat.completions.create(
        model=settings.model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    data = _safe_json_loads(content)

    if not data:
        return InvoiceData()

    return InvoiceData(**data)


def process_document_with_ai(text: str) -> dict:
    classification = classify_document(text)

    if not classification.get("is_invoice_like", False):
        return {
            "document_classification": classification,
            "invoice_data": InvoiceData(),
            "normalized_invoice_data": {},
        }

    invoice_data = extract_invoice_data_from_text(text)
    normalized_invoice_data = normalize_invoice_data(invoice_data)

    enriched_invoice_data = enrich_normalized_invoice_data(
        extracted_text=text,
        raw_invoice_data=invoice_data.model_dump(),
        normalized_invoice_data=normalized_invoice_data,
    )

    return {
        "document_classification": classification,
        "invoice_data": invoice_data,
        "normalized_invoice_data": enriched_invoice_data,
    }