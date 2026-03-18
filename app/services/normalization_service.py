import re
from datetime import datetime

from app.services.arabic_cleanup_service import cleanup_vendor_name


ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EASTERN_ARABIC_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def normalize_digits(value):
    if value is None:
        return None

    value = str(value)
    value = value.translate(ARABIC_INDIC_MAP)
    value = value.translate(EASTERN_ARABIC_MAP)
    value = value.replace("٫", ".")
    value = value.replace("٬", ",")
    return value.strip()


def normalize_currency(value):
    if value is None:
        return None

    value = normalize_digits(value).strip()

    compact = value.lower()
    compact = re.sub(r"[\s\.\-_]", "", compact)

    currency_map = {
        "kwd": "KWD",
        "kd": "KWD",
        "dk": "KWD",
        "ديناركويتي": "KWD",
        "kuwaitidinar": "KWD",

        "sar": "SAR",
        "rs": "SAR",
        "ريالسعودي": "SAR",
        "saudiriyal": "SAR",

        "aed": "AED",
        "de": "AED",
        "درهماماراتي": "AED",
        "uaedirham": "AED",

        "qar": "QAR",
        "rq": "QAR",
        "ريالقطري": "QAR",
        "qataririyal": "QAR",
    }

    return currency_map.get(compact, value)


def normalize_amount(value):
    if value is None:
        return None

    value = normalize_digits(value)
    value = value.strip()

    value = re.sub(r"\b(KWD|KD|K\.D\.|SAR|AED|QAR)\b", "", value, flags=re.IGNORECASE)
    value = value.replace("د.ك", "")
    value = value.replace("دك", "")
    value = value.replace("د ك", "")
    value = value.replace("ر.س", "")
    value = value.replace("د.إ", "")
    value = value.replace("ر.ق", "")

    value = value.strip()
    value = re.sub(r"[^0-9,.\-]", "", value)

    if "," in value and "." in value:
        value = value.replace(",", "")
    elif value.count(",") == 1 and "." not in value:
        value = value.replace(",", ".")

    value = value.strip()

    if value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def normalize_date(value):
    if value is None:
        return None

    value = normalize_digits(value).strip()
    value = value.replace("AM", "").replace("PM", "").strip()
    value = value.replace("am", "").replace("pm", "").strip()
    value = re.sub(r"\s+", " ", value)

    date_formats = [
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%d/%m/%y",
        "%d-%m-%y",
    ]

    for fmt in date_formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    patterns = [
        r"(\d{1,2}/\d{1,2}/\d{4})",
        r"(\d{4}-\d{1,2}-\d{1,2})",
        r"(\d{1,2}-\d{1,2}-\d{4})",
        r"(\d{4}/\d{1,2}/\d{1,2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            extracted = match.group(1)
            for fmt in ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"]:
                try:
                    dt = datetime.strptime(extracted, fmt)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue

    return value


def normalize_invoice_data(invoice_data):
    data = invoice_data.model_dump()

    data["vendor_name"] = cleanup_vendor_name(data.get("vendor_name"))
    data["invoice_date"] = normalize_date(data.get("invoice_date"))
    data["currency"] = normalize_currency(data.get("currency"))

    data["subtotal"] = normalize_amount(data.get("subtotal"))
    data["tax"] = normalize_amount(data.get("tax"))
    data["discount"] = normalize_amount(data.get("discount"))
    data["total"] = normalize_amount(data.get("total"))

    items = data.get("items", [])
    normalized_items = []

    for item in items:
        normalized_items.append({
            "description": item.get("description"),
            "quantity": normalize_amount(item.get("quantity")),
            "unit_price": normalize_amount(item.get("unit_price")),
            "total_price": normalize_amount(item.get("total_price")),
        })

    data["items"] = normalized_items

    return data