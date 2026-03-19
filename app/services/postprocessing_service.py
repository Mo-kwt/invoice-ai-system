import re
from copy import deepcopy

from app.services.normalization_service import (
    normalize_currency,
    normalize_date,
    normalize_digits,
    normalize_amount,
)
from app.services.arabic_cleanup_service import cleanup_vendor_name


INVALID_INVOICE_NUMBER_WORDS = {
    "description",
    "qty",
    "quantity",
    "price",
    "amount",
    "total",
    "item",
    "items",
    "fits",
    "unit",
    "kd",
    "kwd",
    "no",
}

INVOICE_NUMBER_STRONG_LABELS = [
    r"invoice\s*no(?:\.|number)?",
    r"invoice\s*number",
    r"inv(?:\.|oice)?\s*no(?:\.|number)?",
    r"inv(?:\.|oice)?",
    r"رقم\s*الفاتورة",
    r"فاتورة",
]

INVOICE_NUMBER_WEAK_LABELS = [
    r"\bno\b",
]

INVOICE_NUMBER_NEGATIVE_WORDS = {
    "tel",
    "telephone",
    "phone",
    "mobile",
    "customer",
    "cust",
    "order",
    "po",
    "p\.o",
    "ref",
    "reference",
    "account",
}

DATE_STRONG_LABELS = [
    r"invoice\s*date",
    r"tax\s*invoice\s*date",
    r"credit\s*invoice\s*date",
    r"تاريخ\s*الفاتورة",
]

DATE_WEAK_LABELS = [
    r"\bdate\b",
    r"\bdt\b",
    r"التاريخ",
    r"تاريخ",
]

DATE_NEGATIVE_LABELS = [
    r"due\s*date",
    r"delivery\s*date",
    r"print\s*date",
    r"ship(?:ping)?\s*date",
    r"posting\s*date",
]


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_digits(text)
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _score_to_confidence(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def _extract_context_window(text: str, start: int, end: int, window: int = 60) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right]


def _normalize_candidate_value(value: str) -> str:
    if value is None:
        return ""
    value = normalize_digits(str(value))
    value = value.strip()
    value = re.sub(r"^[\s:#\-\/]+", "", value)
    value = re.sub(r"[\s:]+$", "", value)
    value = value.strip(".,;")
    return value.strip()


def find_currency_in_text(text: str):
    text = _clean_text(text)
    currency_patterns = [
        r"\bKWD\b",
        r"\bKD\b",
        r"\bK\.D\.?\b",
        r"د\.?\s*ك",
        r"دينار\s+كويتي",
        r"kuwaiti\s+dinar",
    ]
    for pattern in currency_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_currency(match.group(0))
    return None


def _is_valid_invoice_number(candidate: str) -> bool:
    if not candidate:
        return False

    candidate = _normalize_candidate_value(candidate)
    candidate_upper = candidate.upper()

    if candidate.lower() in INVALID_INVOICE_NUMBER_WORDS:
        return False
    if re.fullmatch(r"[A-Z]+", candidate_upper):
        return False
    if len(candidate) < 2:
        return False
    if re.fullmatch(r"\d{8,}", candidate):
        return False

    has_digit = any(ch.isdigit() for ch in candidate)
    has_alpha = any(ch.isalpha() for ch in candidate)

    if not has_digit and not has_alpha:
        return False
    if has_alpha and not has_digit and len(candidate) < 6:
        return False

    return True


def _has_negative_invoice_context(context: str) -> bool:
    lowered = (context or "").lower()
    for word in INVOICE_NUMBER_NEGATIVE_WORDS:
        if re.search(rf"\b{word}\b", lowered):
            return True
    return False


def collect_invoice_number_candidates(text: str):
    text = _clean_text(text)
    if not text:
        return []

    patterns = [
        ("strong_label", "|".join(INVOICE_NUMBER_STRONG_LABELS)),
        ("weak_label", "|".join(INVOICE_NUMBER_WEAK_LABELS)),
    ]

    candidates = []
    seen = set()

    for label_type, label_pattern in patterns:
        pattern = (
            rf"(?P<label>{label_pattern})\s*[:#\-]?\s*"
            rf"(?P<value>[A-Z0-9][A-Z0-9\-\/]{{1,30}})"
        )
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw_value = _normalize_candidate_value(match.group("value"))
            if not _is_valid_invoice_number(raw_value):
                continue

            dedupe_key = (
                raw_value.lower(),
                match.group("label").lower(),
                match.start("value"),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            context = _extract_context_window(text, match.start(), match.end())
            candidates.append(
                {
                    "value": raw_value,
                    "label": match.group("label"),
                    "label_type": label_type,
                    "context": context,
                    "position": match.start("value"),
                    "position_ratio": round(match.start("value") / max(len(text), 1), 4),
                    "score": 0,
                    "confidence": "low",
                    "ambiguous": False,
                    "reasons": [],
                }
            )

    return candidates


def score_invoice_number_candidate(candidate, context=None):
    if not candidate:
        return candidate

    working = deepcopy(candidate)
    value = working.get("value", "")
    candidate_context = context if context is not None else working.get("context", "")
    lowered_context = (candidate_context or "").lower()

    score = 0
    reasons = []

    label_type = working.get("label_type")
    if label_type == "strong_label":
        score += 45
        reasons.append("strong_label")
    elif label_type == "weak_label":
        score += 20
        reasons.append("weak_label")

    has_digit = any(ch.isdigit() for ch in value)
    has_alpha = any(ch.isalpha() for ch in value)

    if has_digit and has_alpha:
        score += 25
        reasons.append("alnum_pattern")
    elif has_digit:
        score += 10
        reasons.append("numeric_only_pattern")

    if re.fullmatch(r"[A-Z]{1,5}[-\/]?\d{2,}", value, flags=re.IGNORECASE):
        score += 12
        reasons.append("common_invoice_format")

    if len(value) <= 18:
        score += 5
        reasons.append("reasonable_length")

    position_ratio = working.get("position_ratio", 1.0)
    if position_ratio <= 0.2:
        score += 12
        reasons.append("early_position")
    elif position_ratio <= 0.4:
        score += 6
        reasons.append("mid_early_position")

    if _has_negative_invoice_context(lowered_context):
        score -= 35
        reasons.append("negative_context")

    if value.lower() in INVALID_INVOICE_NUMBER_WORDS:
        score -= 50
        reasons.append("invalid_word")

    score = _clamp_score(score)
    working["score"] = score
    working["confidence"] = _score_to_confidence(score)
    working["reasons"] = reasons
    return working


def select_best_invoice_number(candidates):
    scored_candidates = [score_invoice_number_candidate(c) for c in candidates]
    scored_candidates = sorted(
        scored_candidates,
        key=lambda c: (c.get("score", 0), -c.get("position", 10**9)),
        reverse=True,
    )

    if not scored_candidates:
        return {
            "value": None,
            "confidence": "low",
            "candidates": [],
            "selected_candidate": None,
            "ambiguous": False,
        }

    best = scored_candidates[0]
    ambiguous = False

    if len(scored_candidates) > 1:
        second = scored_candidates[1]
        if (
            best.get("score", 0) >= 40
            and second.get("score", 0) >= 40
            and abs(best.get("score", 0) - second.get("score", 0)) <= 8
            and best.get("value") != second.get("value")
        ):
            ambiguous = True

    for candidate in scored_candidates:
        candidate["ambiguous"] = ambiguous

    return {
        "value": None if ambiguous else best.get("value"),
        "confidence": "low" if ambiguous else best.get("confidence", "low"),
        "candidates": scored_candidates,
        "selected_candidate": None if ambiguous else best,
        "ambiguous": ambiguous,
    }


def _is_likely_negative_date_label(label: str) -> bool:
    lowered = (label or "").lower()
    for pattern in DATE_NEGATIVE_LABELS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return True
    return False


def _normalize_date_candidate_value(value: str):
    cleaned = _normalize_candidate_value(value)
    normalized = normalize_date(cleaned)
    return cleaned, normalized


def collect_date_candidates(text: str):
    text = _clean_text(text)
    if not text:
        return []

    date_value_pattern = (
        r"(?P<value>"
        r"(?:[0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})"
        r"|"
        r"(?:[0-9]{4}[\/\-][0-9]{1,2}[\/\-][0-9]{1,2})"
        r")"
    )

    label_patterns = [
        ("strong_label", "|".join(DATE_STRONG_LABELS)),
        ("weak_label", "|".join(DATE_WEAK_LABELS + DATE_NEGATIVE_LABELS)),
    ]

    candidates = []
    seen = set()

    for label_type, label_pattern in label_patterns:
        pattern = rf"(?P<label>{label_pattern})\s*[:\-]?\s*{date_value_pattern}"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw_value, normalized_value = _normalize_date_candidate_value(match.group("value"))
            if not raw_value:
                continue

            dedupe_key = (
                raw_value,
                normalized_value or "",
                match.group("label").lower(),
                match.start("value"),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            context = _extract_context_window(text, match.start(), match.end())
            candidates.append(
                {
                    "value": raw_value,
                    "normalized_value": normalized_value,
                    "label": match.group("label"),
                    "label_type": label_type,
                    "context": context,
                    "position": match.start("value"),
                    "position_ratio": round(match.start("value") / max(len(text), 1), 4),
                    "score": 0,
                    "confidence": "low",
                    "ambiguous": False,
                    "reasons": [],
                }
            )

    generic_pattern = date_value_pattern
    for match in re.finditer(generic_pattern, text, flags=re.IGNORECASE):
        raw_value, normalized_value = _normalize_date_candidate_value(match.group("value"))
        if not raw_value:
            continue

        context = _extract_context_window(text, match.start(), match.end())
        context_lower = context.lower()
        if not any(
            re.search(p, context_lower, flags=re.IGNORECASE)
            for p in DATE_STRONG_LABELS + DATE_WEAK_LABELS + DATE_NEGATIVE_LABELS
        ):
            continue

        dedupe_key = (raw_value, normalized_value or "", "generic", match.start())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        candidates.append(
            {
                "value": raw_value,
                "normalized_value": normalized_value,
                "label": "generic",
                "label_type": "generic",
                "context": context,
                "position": match.start(),
                "position_ratio": round(match.start() / max(len(text), 1), 4),
                "score": 0,
                "confidence": "low",
                "ambiguous": False,
                "reasons": [],
            }
        )

    return candidates


def score_date_candidate(candidate, context=None):
    if not candidate:
        return candidate

    working = deepcopy(candidate)
    candidate_context = context if context is not None else working.get("context", "")
    score = 0
    reasons = []

    label = working.get("label", "")
    label_type = working.get("label_type")

    if _is_likely_negative_date_label(label):
        score -= 50
        reasons.append("negative_label")
    elif label_type == "strong_label":
        score += 45
        reasons.append("strong_label")
    elif label_type == "weak_label":
        score += 18
        reasons.append("weak_label")
    elif label_type == "generic":
        score += 8
        reasons.append("generic_context")

    if working.get("normalized_value"):
        score += 20
        reasons.append("normalizable_date")
    else:
        score -= 20
        reasons.append("unparseable_date")

    position_ratio = working.get("position_ratio", 1.0)
    if position_ratio <= 0.25:
        score += 10
        reasons.append("early_position")
    elif position_ratio <= 0.45:
        score += 5
        reasons.append("mid_early_position")

    lowered_context = (candidate_context or "").lower()
    if any(re.search(pattern, lowered_context, flags=re.IGNORECASE) for pattern in DATE_NEGATIVE_LABELS):
        score -= 25
        reasons.append("negative_context")

    score = _clamp_score(score)
    working["score"] = score
    working["confidence"] = _score_to_confidence(score)
    working["reasons"] = reasons
    return working


def select_best_date(candidates):
    scored_candidates = [score_date_candidate(c) for c in candidates]
    scored_candidates = sorted(
        scored_candidates,
        key=lambda c: (c.get("score", 0), -c.get("position", 10**9)),
        reverse=True,
    )

    if not scored_candidates:
        return {
            "value": None,
            "confidence": "low",
            "candidates": [],
            "selected_candidate": None,
            "ambiguous": False,
        }

    best = scored_candidates[0]
    ambiguous = False

    viable = [
        c for c in scored_candidates
        if c.get("score", 0) >= 35 and c.get("normalized_value")
    ]

    unique_values = []
    for item in viable:
        normalized_value = item.get("normalized_value") or item.get("value")
        if normalized_value not in unique_values:
            unique_values.append(normalized_value)

    if len(unique_values) > 1:
        best_viable = viable[0]
        second_viable = None
        for item in viable[1:]:
            candidate_value = item.get("normalized_value") or item.get("value")
            best_value = best_viable.get("normalized_value") or best_viable.get("value")
            if candidate_value != best_value:
                second_viable = item
                break

        if (
            second_viable
            and abs(best_viable.get("score", 0) - second_viable.get("score", 0)) <= 10
        ):
            ambiguous = True

    for candidate in scored_candidates:
        candidate["ambiguous"] = ambiguous

    selected_value = best.get("normalized_value") or best.get("value")
    return {
        "value": None if ambiguous else selected_value,
        "confidence": "low" if ambiguous else best.get("confidence", "low"),
        "candidates": scored_candidates,
        "selected_candidate": None if ambiguous else best,
        "ambiguous": ambiguous,
    }


def build_postprocessing_debug_info(
    extracted_text: str,
    raw_invoice_data: dict | None = None,
    normalized_invoice_data: dict | None = None,
):
    del raw_invoice_data, normalized_invoice_data

    invoice_number_selection = select_best_invoice_number(
        collect_invoice_number_candidates(extracted_text)
    )
    invoice_date_selection = select_best_date(
        collect_date_candidates(extracted_text)
    )

    return {
        "invoice_number_candidates": invoice_number_selection.get("candidates", []),
        "invoice_date_candidates": invoice_date_selection.get("candidates", []),
        "selected_candidate": {
            "invoice_number": invoice_number_selection.get("selected_candidate"),
            "invoice_date": invoice_date_selection.get("selected_candidate"),
        },
        "ambiguity": {
            "invoice_number": invoice_number_selection.get("ambiguous", False),
            "invoice_date": invoice_date_selection.get("ambiguous", False),
        },
    }


def find_date_in_text(text: str):
    result = select_best_date(collect_date_candidates(text))
    return result.get("value")


def find_invoice_number_in_text(text: str):
    result = select_best_invoice_number(collect_invoice_number_candidates(text))
    return result.get("value")


def find_total_in_text(text: str):
    text = _clean_text(text)
    labeled_patterns = [
        r"(?:grand\s*total|total|totl|net|amount|الإجمالي|المجموع|الصافي)\s*[:\-]?\s*([0-9][0-9,\.\s]*)",
    ]
    for pattern in labeled_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for candidate in matches:
            amount = normalize_amount(candidate)
            if amount is not None:
                return amount

    currency_amount_patterns = [
        r"([0-9][0-9,\.\s]*)\s*(?:KWD|KD|K\.D\.?)",
        r"(?:KWD|KD|K\.D\.?)\s*([0-9][0-9,\.\s]*)",
    ]
    for pattern in currency_amount_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for candidate in matches:
            amount = normalize_amount(candidate)
            if amount is not None:
                return amount

    return None


def find_vendor_name_in_weak_text(text: str):
    text = _clean_text(text)
    if not text:
        return None

    lowered = text.lower()

    stripped = re.sub(
        r"\b(inv|invoice|no|date|dt|itm|item|total|totl|kd|kwd)\b[:\-]?",
        " ",
        lowered,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(r"\s+", " ", stripped).strip()

    match = re.search(
        r"^(.*?)(?:\bno\b|\bdt\b|[0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})",
        stripped,
        flags=re.IGNORECASE,
    )
    candidate = match.group(1).strip() if match else stripped

    if not candidate:
        return None

    candidate = re.sub(r"\b(inv|invoice)\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    if len(candidate) < 5:
        return None

    return candidate.title()


def remove_empty_items(data: dict):
    items = data.get("items", [])
    cleaned_items = []
    for item in items:
        if not item:
            continue

        has_real_value = any(
            item.get(field) not in (None, "", 0, 0.0)
            for field in ["description", "quantity", "unit_price", "total_price"]
        )
        if has_real_value:
            cleaned_items.append(item)

    data["items"] = cleaned_items
    return data


def enrich_normalized_invoice_data(
    extracted_text: str,
    raw_invoice_data: dict,
    normalized_invoice_data: dict,
):
    result = deepcopy(normalized_invoice_data)

    result["vendor_name"] = cleanup_vendor_name(result.get("vendor_name"))
    if not result.get("vendor_name"):
        weak_vendor_name = find_vendor_name_in_weak_text(extracted_text)
        if weak_vendor_name:
            result["vendor_name"] = cleanup_vendor_name(weak_vendor_name)

    raw_currency = raw_invoice_data.get("currency")
    if raw_currency:
        result["currency"] = normalize_currency(raw_currency)

    if not result.get("currency"):
        result["currency"] = find_currency_in_text(extracted_text)

    raw_date = raw_invoice_data.get("invoice_date")
    if raw_date:
        result["invoice_date"] = normalize_date(raw_date)

    if not result.get("invoice_date"):
        result["invoice_date"] = find_date_in_text(extracted_text)

    if not result.get("invoice_number"):
        result["invoice_number"] = find_invoice_number_in_text(extracted_text)

    if result.get("total") is None:
        result["total"] = find_total_in_text(extracted_text)

    result = remove_empty_items(result)

    return result