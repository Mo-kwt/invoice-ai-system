import json
from openai import OpenAI

from app.config import settings
from app.prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT_TEMPLATE,
)

client = OpenAI(api_key=settings.openai_api_key)

INVOICE_LIKE_TYPES = {
    "invoice",
    "tax_invoice",
    "proforma_invoice",
    "credit_note",
    "debit_note",
}


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def classify_document(document_text: str):
    if not document_text or not document_text.strip():
        return {
            "document_type": "unknown",
            "invoice_likelihood": 0.0,
            "is_invoice_like": False,
            "reason": "النص فارغ",
            "key_evidence": []
        }

    prompt = CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
        document_text=document_text[:12000]
    )

    response = client.chat.completions.create(
        model=settings.model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    data = _safe_json_loads(content)

    if not data:
        return {
            "document_type": "unknown",
            "invoice_likelihood": 0.0,
            "is_invoice_like": False,
            "reason": "تعذر قراءة نتيجة التصنيف",
            "key_evidence": []
        }

    document_type = str(data.get("document_type", "unknown")).strip().lower()
    invoice_likelihood = data.get("invoice_likelihood", 0.0)
    reason = data.get("reason", "")
    key_evidence = data.get("key_evidence", [])

    # القرار النهائي عندنا نحن، وليس عند النموذج
    is_invoice_like = document_type in INVOICE_LIKE_TYPES

    return {
        "document_type": document_type,
        "invoice_likelihood": invoice_likelihood,
        "is_invoice_like": is_invoice_like,
        "reason": reason,
        "key_evidence": key_evidence,
    }