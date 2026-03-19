import json
import re
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


INVOICE_NUMBER_STOP_TOKENS = [
    "date",
    "invoice date",
    "item",
    "items",
    "qty",
    "quantity",
    "unit",
    "price",
    "total",
    "subtotal",
    "tax",
    "discount",
    "vendor",
    "bill to",
    "ship to",
    "description",
    "التاريخ",
    "تاريخ",
    "الصنف",
    "الأصناف",
    "الكمية",
    "السعر",
    "الإجمالي",
    "اجمالي",
    "المبلغ",
    "الوصف",
]

DATE_STOP_TOKENS = [
    "item",
    "items",
    "qty",
    "quantity",
    "unit",
    "price",
    "total",
    "subtotal",
    "tax",
    "discount",
    "vendor",
    "invoice no",
    "invoice number",
    "رقم الفاتورة",
    "الصنف",
    "الكمية",
    "السعر",
    "الإجمالي",
]


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _has_meaningful_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def _merge_invoice_data(
    base_data: dict,
    fallback_data: dict,
    prefer_override_fields: set[str] | None = None,
) -> dict:
    merged = dict(base_data)
    prefer_override_fields = prefer_override_fields or set()

    for key, value in fallback_data.items():
        base_value = merged.get(key)

        if key in prefer_override_fields:
            if _has_meaningful_value(value):
                merged[key] = value
            continue

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

    strong_signal_count = sum(
        [
            1 if english_hits >= 2 else 0,
            1 if arabic_hits >= 2 else 0,
            1 if has_total_like else 0,
            1 if has_company_like else 0,
            1 if has_date_like else 0,
            1 if has_phone_like else 0,
            1 if arabic_count >= 10 else 0,
            1 if alpha_count >= 30 else 0,
        ]
    )

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
        "inv",
        "dt",
        "itm",
        "totl",
        "no",
        "المبلغ",
        "مؤسسة",
        "شركة",
        "التوقيع",
        "السادة",
        "الحساب",
        "total",
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


def _normalize_text_for_evidence(text: str) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_lines_for_evidence(text: str) -> list[str]:
    normalized = _normalize_text_for_evidence(text)
    lines = [line.strip() for line in normalized.split("\n")]
    return [line for line in lines if line]


def _clean_candidate(value: str) -> str:
    value = (value or "").strip()
    value = value.strip(":：#-–|")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _truncate_on_stop_tokens(value: str, stop_tokens: list[str]) -> str:
    candidate = value or ""
    lowered = candidate.lower()

    cut_positions = []
    for token in stop_tokens:
        idx = lowered.find(token.lower())
        if idx > 0:
            cut_positions.append(idx)

    if cut_positions:
        candidate = candidate[: min(cut_positions)]

    return _clean_candidate(candidate)


def _is_probable_phone(value: str) -> bool:
    digits = re.sub(r"\D", "", value or "")
    return 7 <= len(digits) <= 12 and ("+" in (value or "") or len(digits) >= 8)


def _is_probable_money(value: str) -> bool:
    lower = (value or "").lower()
    money_tokens = ["kwd", "usd", "sar", "aed", "ريال", "دينار", "د.ك", "kd"]
    return any(token in lower for token in money_tokens)


def _looks_like_invoice_number_candidate(value: str) -> bool:
    value = _clean_candidate(value)
    if not value:
        return False

    if len(value) < 3 or len(value) > 40:
        return False

    value = _truncate_on_stop_tokens(value, INVOICE_NUMBER_STOP_TOKENS)
    if not value:
        return False

    if _is_probable_phone(value):
        return False

    if _is_probable_money(value):
        return False

    lower = value.lower()
    weak_exact = {
        "invoice",
        "inv",
        "date",
        "qty",
        "total",
        "subtotal",
        "amount",
        "cash",
        "receipt",
        "فاتورة",
        "التاريخ",
    }
    if lower in weak_exact:
        return False

    digits_only = re.sub(r"\D", "", value)
    has_digit = any(ch.isdigit() for ch in value)
    has_alpha = any(ch.isalpha() for ch in value)
    has_separator = bool(re.search(r"[-_/]", value))

    if value.isdigit():
        return len(digits_only) >= 6

    if has_digit and (has_alpha or has_separator or len(digits_only) >= 6):
        return True

    return False


def _extract_candidate_after_indicator(line: str, indicator: str, stop_tokens: list[str] | None = None) -> str | None:
    lower_line = line.lower()
    idx = lower_line.find(indicator.lower())
    if idx == -1:
        return None

    tail = line[idx + len(indicator):].strip()
    tail = re.sub(r"^[\s:：#\-–|]+", "", tail).strip()
    if not tail:
        return None

    match = re.match(
        r"([A-Za-z0-9\u0600-\u06FF\-_\/\.]+(?:\s+[A-Za-z0-9\u0600-\u06FF\-_\/\.]+){0,4})",
        tail,
    )
    value = match.group(1) if match else tail

    if stop_tokens:
        value = _truncate_on_stop_tokens(value, stop_tokens)

    return _clean_candidate(value)


def _score_invoice_number_candidate(value: str, line: str, indicator_hit: bool) -> int:
    score = 0
    clean = _clean_candidate(value)
    lower_line = line.lower()

    if indicator_hit:
        score += 5

    if any(ch.isdigit() for ch in clean):
        score += 3

    if any(ch.isalpha() for ch in clean):
        score += 1

    if re.search(r"[-_/]", clean):
        score += 2

    digit_count = len(re.sub(r"\D", "", clean))
    if 3 <= digit_count <= 16:
        score += 2

    if "invoice" in lower_line or "inv" in lower_line or "فاتورة" in line:
        score += 2

    if _is_probable_phone(clean):
        score -= 5

    if _is_probable_money(clean):
        score -= 4

    if clean.isdigit():
        score -= 3

    return score


def _extract_invoice_number_candidates(text: str) -> list[dict]:
    lines = _split_lines_for_evidence(text)
    candidates: list[dict] = []

    indicators = [
        "رقم الفاتورة",
        "رقم فاتورة",
        "فاتورة رقم",
        "الفاتورة رقم",
        "invoice no",
        "invoice number",
        "invoice #",
        "invoice#",
        "inv no",
        "inv number",
        "inv #",
        "bill no",
        "bill number",
        "receipt no",
        "receipt number",
    ]

    for line in lines:
        line_lower = line.lower()

        for indicator in indicators:
            if indicator.lower() in line_lower:
                value = _extract_candidate_after_indicator(
                    line,
                    indicator,
                    stop_tokens=INVOICE_NUMBER_STOP_TOKENS,
                )
                if value and _looks_like_invoice_number_candidate(value):
                    candidates.append(
                        {
                            "value": value,
                            "score": _score_invoice_number_candidate(value, line, True),
                            "source_line": line,
                            "reason": f"indicator:{indicator}",
                        }
                    )

        generic_matches = re.findall(
            r"\b(?:[A-Za-z]{1,6}[-_/]?\d{2,}[A-Za-z0-9\-_/]*|\d{6,})\b",
            line,
        )
        for match in generic_matches:
            cleaned = _truncate_on_stop_tokens(match, INVOICE_NUMBER_STOP_TOKENS)
            if cleaned and _looks_like_invoice_number_candidate(cleaned):
                candidates.append(
                    {
                        "value": _clean_candidate(cleaned),
                        "score": _score_invoice_number_candidate(cleaned, line, False),
                        "source_line": line,
                        "reason": "generic_pattern",
                    }
                )

    deduped: dict[str, dict] = {}
    for candidate in candidates:
        key = candidate["value"].strip().lower()
        existing = deduped.get(key)
        if existing is None or candidate["score"] > existing["score"]:
            deduped[key] = candidate

    result = list(deduped.values())
    result.sort(key=lambda x: x["score"], reverse=True)
    return result[:10]


def _looks_like_date_candidate(value: str) -> bool:
    value = _clean_candidate(value)
    if not value:
        return False

    value = _truncate_on_stop_tokens(value, DATE_STOP_TOKENS)

    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b",
    ]
    if any(re.search(pattern, value) for pattern in patterns):
        return True

    month_words = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec",
        "january", "february", "march", "april", "june", "july",
        "august", "september", "october", "november", "december",
    ]
    lower = value.lower()
    return any(month in lower for month in month_words) and bool(re.search(r"\d", value))


def _extract_date_candidate_from_line(line: str, indicator: str | None = None) -> str | None:
    if indicator:
        lower_line = line.lower()
        idx = lower_line.find(indicator.lower())
        if idx != -1:
            tail = line[idx + len(indicator):].strip()
            tail = re.sub(r"^[\s:：#\-–|]+", "", tail).strip()
            tail = _truncate_on_stop_tokens(tail, DATE_STOP_TOKENS)

            date_match = re.search(
                r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b)",
                tail,
            )
            if date_match:
                return _clean_candidate(date_match.group(1))

    date_match = re.search(
        r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b)",
        line,
    )
    if date_match:
        return _clean_candidate(date_match.group(1))

    return None


def _score_date_candidate(value: str, line: str, indicator_hit: bool) -> int:
    score = 0
    lower_line = line.lower()

    if indicator_hit:
        score += 5

    if re.search(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", value):
        score += 3
    elif re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", value):
        score += 2

    if "date" in lower_line or "تاريخ" in line:
        score += 2

    if "due" in lower_line:
        score -= 2

    return score


def _extract_invoice_date_candidates(text: str) -> list[dict]:
    lines = _split_lines_for_evidence(text)
    candidates: list[dict] = []

    indicators = [
        "تاريخ الفاتورة",
        "تاريخ",
        "التاريخ",
        "invoice date",
        "date",
        "issued on",
        "issue date",
    ]

    for line in lines:
        line_lower = line.lower()

        for indicator in indicators:
            if indicator.lower() in line_lower:
                value = _extract_date_candidate_from_line(line, indicator)
                if value and _looks_like_date_candidate(value):
                    candidates.append(
                        {
                            "value": value,
                            "score": _score_date_candidate(value, line, True),
                            "source_line": line,
                            "reason": f"indicator:{indicator}",
                        }
                    )

        generic = _extract_date_candidate_from_line(line)
        if generic and _looks_like_date_candidate(generic):
            candidates.append(
                {
                    "value": generic,
                    "score": _score_date_candidate(generic, line, False),
                    "source_line": line,
                    "reason": "generic_pattern",
                }
            )

    deduped: dict[str, dict] = {}
    for candidate in candidates:
        key = candidate["value"].strip().lower()
        existing = deduped.get(key)
        if existing is None or candidate["score"] > existing["score"]:
            deduped[key] = candidate

    result = list(deduped.values())
    result.sort(key=lambda x: x["score"], reverse=True)
    return result[:10]


def _detect_items_table(text: str) -> dict:
    lines = _split_lines_for_evidence(text)

    item_table_hints = [
        "item",
        "items",
        "description",
        "qty",
        "quantity",
        "unit",
        "price",
        "amount",
        "total",
        "product",
        "service",
        "الصنف",
        "الأصناف",
        "البيان",
        "الوصف",
        "الكمية",
        "سعر",
        "السعر",
        "الإجمالي",
        "اجمالي",
        "المبلغ",
        "الوحدة",
    ]

    scored_lines = []
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        hit_tokens = [token for token in item_table_hints if token in line_lower or token in line]
        if not hit_tokens:
            continue

        digit_count = len(re.findall(r"\d", line))
        separator_count = len(re.findall(r"[|:\-]", line))
        score = len(hit_tokens) * 2 + min(digit_count, 4) + min(separator_count, 2)

        scored_lines.append(
            {
                "line_index": idx,
                "line": line,
                "tokens": hit_tokens,
                "score": score,
            }
        )

    scored_lines.sort(key=lambda x: x["score"], reverse=True)
    top_lines = scored_lines[:5]

    probable = False
    confidence = "low"

    if len([x for x in scored_lines if x["score"] >= 5]) >= 2:
        probable = True
        confidence = "high"
    elif len([x for x in scored_lines if x["score"] >= 5]) == 1 or len(scored_lines) >= 2:
        probable = True
        confidence = "medium"

    return {
        "items_table_detected": probable,
        "items_table_confidence": confidence,
        "items_table_lines": top_lines,
    }


def _build_field_evidence(
    text: str,
    normalized_invoice_data: dict,
) -> dict:
    invoice_number_candidates = _extract_invoice_number_candidates(text)
    invoice_date_candidates = _extract_invoice_date_candidates(text)
    items_info = _detect_items_table(text)

    suggested_invoice_number = invoice_number_candidates[0]["value"] if invoice_number_candidates else None
    suggested_invoice_date = invoice_date_candidates[0]["value"] if invoice_date_candidates else None

    invoice_number_confidence = "low"
    if invoice_number_candidates:
        if len(invoice_number_candidates) == 1 and invoice_number_candidates[0]["score"] >= 8:
            invoice_number_confidence = "high"
        elif invoice_number_candidates[0]["score"] >= 7:
            invoice_number_confidence = "medium"

    invoice_date_confidence = "low"
    if invoice_date_candidates:
        if len(invoice_date_candidates) == 1 and invoice_date_candidates[0]["score"] >= 8:
            invoice_date_confidence = "high"
        elif invoice_date_candidates[0]["score"] >= 6:
            invoice_date_confidence = "medium"

    extracted_invoice_number = (normalized_invoice_data.get("invoice_number") or "").strip()
    extracted_invoice_date = (normalized_invoice_data.get("invoice_date") or "").strip()
    extracted_items = normalized_invoice_data.get("items") or []

    field_review_flags = []

    normalized_candidate_values = {
        (candidate.get("value") or "").strip()
        for candidate in invoice_number_candidates
    }

    extracted_invoice_number_matches_candidate = (
        bool(extracted_invoice_number)
        and extracted_invoice_number in normalized_candidate_values
    )

    if not extracted_invoice_number:
        if suggested_invoice_number:
            field_review_flags.append("invoice_number_missing_but_candidate_exists")
        else:
            field_review_flags.append("invoice_number_missing")
    else:
        if (
            suggested_invoice_number
            and extracted_invoice_number != suggested_invoice_number
            and not extracted_invoice_number_matches_candidate
        ):
            field_review_flags.append("invoice_number_differs_from_top_candidate")

        if invoice_number_confidence == "low" and not extracted_invoice_number_matches_candidate:
            field_review_flags.append("invoice_number_low_confidence")

        strong_distinct_candidates = {
            c["value"]
            for c in invoice_number_candidates
            if c.get("score", 0) >= 8
        }
        if len(strong_distinct_candidates) >= 3 and not extracted_invoice_number_matches_candidate:
            field_review_flags.append("invoice_number_ambiguous")

    if not extracted_invoice_date:
        if suggested_invoice_date:
            field_review_flags.append("invoice_date_missing_but_candidate_exists")
        else:
            field_review_flags.append("invoice_date_missing")
    else:
        if suggested_invoice_date and extracted_invoice_date != suggested_invoice_date:
            field_review_flags.append("invoice_date_differs_from_top_candidate")
        if invoice_date_confidence == "low":
            field_review_flags.append("invoice_date_low_confidence")
        if len(invoice_date_candidates) >= 3:
            field_review_flags.append("invoice_date_ambiguous")

    if not extracted_items:
        if items_info["items_table_detected"]:
            field_review_flags.append("items_missing_but_table_detected")
        else:
            field_review_flags.append("items_not_reliably_detected")
    else:
        if items_info["items_table_confidence"] == "low":
            field_review_flags.append("items_low_confidence")

    return {
        "invoice_number_candidates": invoice_number_candidates,
        "invoice_date_candidates": invoice_date_candidates,
        "suggested_invoice_number": suggested_invoice_number,
        "suggested_invoice_number_confidence": invoice_number_confidence,
        "suggested_invoice_date": suggested_invoice_date,
        "suggested_invoice_date_confidence": invoice_date_confidence,
        "items_table_detected": items_info["items_table_detected"],
        "items_table_confidence": items_info["items_table_confidence"],
        "items_table_lines": items_info["items_table_lines"],
        "field_review_flags": sorted(set(field_review_flags)),
        "evidence_summary": {
            "invoice_number_candidates_count": len(invoice_number_candidates),
            "invoice_date_candidates_count": len(invoice_date_candidates),
            "items_table_detected": items_info["items_table_detected"],
            "items_table_confidence": items_info["items_table_confidence"],
        },
    }


def extract_invoice_data_from_text(text: str) -> InvoiceData:
    if not text or not text.strip():
        return InvoiceData()

    prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(document_text=text[:12000])

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
- Extract only these fields: vendor_name, invoice_number, invoice_date, subtotal, tax, discount, total, currency, items

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
                "field_evidence": {
                    "invoice_number_candidates": [],
                    "invoice_date_candidates": [],
                    "suggested_invoice_number": None,
                    "suggested_invoice_number_confidence": "low",
                    "suggested_invoice_date": None,
                    "suggested_invoice_date_confidence": "low",
                    "items_table_detected": False,
                    "items_table_confidence": "low",
                    "items_table_lines": [],
                    "field_review_flags": [],
                    "evidence_summary": {
                        "invoice_number_candidates_count": 0,
                        "invoice_date_candidates_count": 0,
                        "items_table_detected": False,
                        "items_table_confidence": "low",
                    },
                },
                "field_review_flags": [],
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

        print("DEBUG_VISION_RESULT:", vision_invoice_data.model_dump())

        merged_data = _merge_invoice_data(
            base_data=final_invoice_data.model_dump(),
            fallback_data=vision_invoice_data.model_dump(),
            prefer_override_fields={"total"},
        )

        final_invoice_data = InvoiceData(**merged_data)
        final_normalized_data = normalize_invoice_data(final_invoice_data)
        used_vision_fallback = True

    enriched_invoice_data = enrich_normalized_invoice_data(
        extracted_text=text,
        raw_invoice_data=final_invoice_data.model_dump(),
        normalized_invoice_data=final_normalized_data,
    )

    field_evidence = _build_field_evidence(
        text=text,
        normalized_invoice_data=enriched_invoice_data,
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
        "field_evidence": field_evidence,
        "field_review_flags": field_evidence.get("field_review_flags", []),
    }

    return {
        "document_classification": classification,
        "invoice_data": final_invoice_data,
        "normalized_invoice_data": enriched_invoice_data,
        "debug_info": debug_info,
    }