import json
from pathlib import Path
from typing import Optional

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
from app.services.vision_extraction_service import extract_invoice_data_from_image

client = OpenAI(api_key=settings.openai_api_key)


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _merge_invoice_data(base_data: dict, fallback_data: dict) -> dict:
    merged = dict(base_data)

    for key, value in fallback_data.items():
        base_value = merged.get(key)
        if base_value in (None, "", []):
            merged[key] = value

    return merged


def _find_missing_fields(current_data: dict) -> list[str]:
    missing_fields = []

    if not current_data.get("vendor_name"):
        missing_fields.append("vendor_name")
    if not current_data.get("invoice_number"):
        missing_fields.append("invoice_number")
    if not current_data.get("invoice_date"):
        missing_fields.append("invoice_date")
    if current_data.get("total") is None:
        missing_fields.append("total")
    if not current_data.get("currency"):
        missing_fields.append("currency")

    return missing_fields


def _is_weak_vendor_name(vendor_name: Optional[str]) -> bool:
    if not vendor_name:
        return True

    vendor_name = vendor_name.strip()

    if len(vendor_name) < 6:
        return True

    weak_tokens = {
        "inv",
        "invoice",
        "no",
        "dt",
        "item",
        "itm",
        "total",
        "totl",
        "kd",
        "kwd",
        "مبلغ",
        "المبلغ",
        "السادة",
        "التوقيع",
    }

    if vendor_name.lower() in weak_tokens:
        return True

    return False


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    lowered = (text or "").lower()
    return sum(1 for token in keywords if token.lower() in lowered)


def _looks_like_weak_invoice_text(text: str) -> bool:
    clean_text = (text or "").strip()
    lowered_text = clean_text.lower()

    if not clean_text:
        return False

    english_tokens = [
        "invoice",
        "inv",
        "total",
        "subtotal",
        "tax",
        "discount",
        "amount",
        "balance",
        "rate",
        "qty",
        "item",
        "no",
        "dt",
        "kd",
        "kwd",
        "sign",
        "receiver",
        "receivers sign",
    ]

    arabic_tokens = [
        "فاتورة",
        "مؤسسة",
        "شركة",
        "المبلغ",
        "اجمالي",
        "الإجمالي",
        "المجموع",
        "السادة",
        "الحساب",
        "التوقيع",
        "استلم",
        "رقم",
        "التاريخ",
        "الكمية",
        "الصنف",
    ]

    english_hits = _count_keyword_hits(lowered_text, english_tokens)
    arabic_hits = _count_keyword_hits(clean_text, arabic_tokens)

    digit_count = sum(ch.isdigit() for ch in clean_text)
    arabic_count = sum(1 for ch in clean_text if "\u0600" <= ch <= "\u06FF")
    alpha_count = sum(ch.isalpha() for ch in clean_text)

    has_total_like = (
        "total" in lowered_text
        or "المبلغ" in clean_text
        or "اجمالي" in clean_text
        or "الإجمالي" in clean_text
    )

    has_company_like = (
        "مؤسسة" in clean_text
        or "شركة" in clean_text
        or "est" in lowered_text
    )

    has_date_like = "/" in clean_text or "-" in clean_text
    has_phone_like = digit_count >= 7

    strong_signal_count = sum([
        1 if english_hits >= 2 else 0,
        1 if arabic_hits >= 2 else 0,
        1 if has_total_like else 0,
        1 if has_company_like else 0,
        1 if has_date_like else 0,
        1 if has_phone_like else 0,
        1 if arabic_count >= 10 else 0,
        1 if alpha_count >= 30 else 0,
    ])

    return strong_signal_count >= 3


def _should_use_fallback(text: str, normalized_data: dict) -> tuple[bool, list[str], dict]:
    reasons = []
    clean_text = (text or "").strip()
    lowered_text = clean_text.lower()
    text_length = len(clean_text)

    missing_fields = _find_missing_fields(normalized_data)
    vendor_name = normalized_data.get("vendor_name")
    weak_vendor = _is_weak_vendor_name(vendor_name)

    if missing_fields:
        reasons.append("missing_required_fields")

    if text_length < 80:
        reasons.append("very_short_extracted_text")
    elif text_length < 120:
        reasons.append("short_extracted_text")
    elif text_length < 500:
        reasons.append("medium_length_ocr_text")

    if weak_vendor:
        reasons.append("weak_vendor_name")

    weak_ocr_tokens = [
        "inv", "dt", "itm", "totl", "no",
        "المبلغ", "مؤسسة", "شركة", "التوقيع", "السادة", "الحساب", "total"
    ]
    weak_hits = sum(1 for token in weak_ocr_tokens if token in lowered_text or token in clean_text)

    if weak_hits >= 2:
        reasons.append("weak_ocr_style_text")

    digit_count = sum(ch.isdigit() for ch in clean_text)
    alpha_count = sum(ch.isalpha() for ch in clean_text)
    arabic_count = sum(1 for ch in clean_text if "\u0600" <= ch <= "\u06FF")

    if alpha_count > 0 and digit_count > 0 and text_length < 100:
        reasons.append("very_compact_mixed_text")

    if text_length < 140 and weak_vendor:
        reasons.append("short_text_with_weak_vendor")

    if text_length >= 140 and text_length < 800 and len(missing_fields) >= 2:
        reasons.append("ocr_text_with_missing_core_fields")

    if arabic_count >= 20 and len(missing_fields) >= 2:
        reasons.append("arabic_ocr_text_with_missing_fields")

    should_fallback = (
        len(missing_fields) > 0
        or text_length < 80
        or (text_length < 100 and "very_compact_mixed_text" in reasons)
        or (text_length < 140 and weak_vendor)
        or (text_length < 800 and len(missing_fields) >= 2)
        or (arabic_count >= 20 and len(missing_fields) >= 2)
    )

    decision_debug = {
        "text_length": text_length,
        "weak_hits": weak_hits,
        "weak_vendor": weak_vendor,
        "vendor_name_before_fallback": vendor_name,
        "missing_fields_count": len(missing_fields),
        "missing_fields_list": missing_fields,
        "alpha_count": alpha_count,
        "digit_count": digit_count,
        "arabic_count": arabic_count,
        "should_fallback": should_fallback,
    }

    return should_fallback, reasons, decision_debug


def _should_use_vision_fallback(
    pdf_path: Optional[str],
    classification: dict,
    normalized_data: dict,
    text: str,
) -> tuple[bool, list[str]]:
    reasons = []

    if not pdf_path:
        return False, reasons

    missing_fields = _find_missing_fields(normalized_data)
    text_length = len((text or "").strip())

    if len(missing_fields) >= 2:
        reasons.append("missing_core_fields_after_text_extraction")

    if classification.get("is_invoice_like", False) and len(missing_fields) >= 2:
        reasons.append("invoice_like_but_text_fields_incomplete")

    if text_length >= 150 and len(missing_fields) >= 2:
        reasons.append("ocr_text_present_but_not_sufficient")

    should_use = len(reasons) > 0
    return should_use, reasons


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

    try:
        return InvoiceData(**data)
    except Exception:
        return InvoiceData()


def extract_missing_fields_from_text(
    text: str,
    current_data: dict,
) -> InvoiceData:
    if not text or not text.strip():
        return InvoiceData()

    missing_fields = _find_missing_fields(current_data)

    focused_prompt = f"""
You are extracting invoice fields from weak OCR text.

Current extracted data:
{json.dumps(current_data, ensure_ascii=False, indent=2)}

Missing fields:
{json.dumps(missing_fields, ensure_ascii=False)}

Instructions:
- Return JSON only.
- This OCR text may be short, noisy, partially damaged, mixed Arabic/English, and contain recognition errors.
- Try to improve weak fields, not only missing fields.
- If vendor_name appears in both Arabic and English, prefer the Arabic official name if it exists clearly in the OCR text.
- The document may still be an invoice even if the OCR text is messy.
- Infer carefully from labels like inv, no, dt, total, kd, kwd, مؤسسة, شركة, المبلغ, الإجمالي, التوقيع.
- Keep existing correct values when possible.
- If a field cannot be found confidently, return null for that field.
- Extract only these fields:
  vendor_name, invoice_number, invoice_date, subtotal, tax, discount, total, currency, items

Document text:
{text[:12000]}
""".strip()

    response = client.chat.completions.create(
        model=settings.model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": focused_prompt},
        ],
    )

    content = response.choices[0].message.content
    data = _safe_json_loads(content)

    if not data:
        return InvoiceData()

    try:
        return InvoiceData(**data)
    except Exception:
        return InvoiceData()


def process_document_with_ai(text: str, pdf_path: Optional[str] = None) -> dict:
    classification = classify_document(text)
    weak_invoice_override = _looks_like_weak_invoice_text(text)

    if not classification.get("is_invoice_like", False) and not weak_invoice_override:
        return {
            "document_classification": classification,
            "invoice_data": InvoiceData(),
            "normalized_invoice_data": {},
            "debug_info": {
                "used_fallback": False,
                "used_vision_fallback": False,
                "missing_fields_before_fallback": [],
                "fallback_reasons": [],
                "vision_fallback_reasons": [],
                "fallback_decision_debug": {},
                "text_length": len(text.strip()) if text else 0,
                "weak_invoice_override": False,
            },
        }

    if weak_invoice_override and not classification.get("is_invoice_like", False):
        classification = {
            **classification,
            "is_invoice_like": True,
            "document_type": "invoice",
            "invoice_likelihood": max(classification.get("invoice_likelihood", 0.0), 0.55),
            "override_reason": "weak_invoice_text_pattern",
        }

    invoice_data = extract_invoice_data_from_text(text)
    normalized_invoice_data = normalize_invoice_data(invoice_data)

    missing_fields_before_fallback = _find_missing_fields(normalized_invoice_data)
    needs_ai_fallback, fallback_reasons, fallback_decision_debug = _should_use_fallback(
        text=text,
        normalized_data=normalized_invoice_data,
    )

    final_invoice_data = invoice_data
    final_normalized_data = normalized_invoice_data
    used_fallback = False
    used_vision_fallback = False
    vision_fallback_reasons = []

    if needs_ai_fallback:
        used_fallback = True

        fallback_invoice_data = extract_missing_fields_from_text(
            text=text,
            current_data=normalized_invoice_data,
        )

        merged_data = _merge_invoice_data(
            base_data=invoice_data.model_dump(),
            fallback_data=fallback_invoice_data.model_dump(),
        )

        final_invoice_data = InvoiceData(**merged_data)
        final_normalized_data = normalize_invoice_data(final_invoice_data)

    should_use_vision, vision_fallback_reasons = _should_use_vision_fallback(
        pdf_path=pdf_path,
        classification=classification,
        normalized_data=final_normalized_data,
        text=text,
    )

    if should_use_vision and pdf_path:
        image_path = str(Path(pdf_path).with_suffix(".png"))

        vision_invoice_data = extract_invoice_data_from_image(
            image_path=image_path,
            current_text=text,
            current_data=final_normalized_data,
        )

        merged_data = _merge_invoice_data(
            base_data=final_invoice_data.model_dump(),
            fallback_data=vision_invoice_data.model_dump(),
        )

        final_invoice_data = InvoiceData(**merged_data)
        final_normalized_data = normalize_invoice_data(final_invoice_data)
        used_vision_fallback = True

    enriched_invoice_data = enrich_normalized_invoice_data(
        extracted_text=text,
        raw_invoice_data=final_invoice_data.model_dump(),
        normalized_invoice_data=final_normalized_data,
    )

    debug_info = {
        "used_fallback": used_fallback,
        "used_vision_fallback": used_vision_fallback,
        "missing_fields_before_fallback": missing_fields_before_fallback,
        "fallback_reasons": fallback_reasons,
        "vision_fallback_reasons": vision_fallback_reasons,
        "fallback_decision_debug": fallback_decision_debug,
        "text_length": len(text.strip()) if text else 0,
        "weak_invoice_override": weak_invoice_override,
    }

    return {
        "document_classification": classification,
        "invoice_data": final_invoice_data,
        "normalized_invoice_data": enriched_invoice_data,
        "debug_info": debug_info,
    }