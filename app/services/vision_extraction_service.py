import base64
import json
from pathlib import Path
from typing import Optional

from openai import OpenAI

from app.config import settings
from app.schemas.invoice import InvoiceData
from app.prompts import EXTRACTION_SYSTEM_PROMPT

client = OpenAI(api_key=settings.openai_api_key)


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_invoice_data_from_image(
    image_path: str,
    current_text: str = "",
    current_data: Optional[dict] = None,
) -> InvoiceData:
    image_file = Path(image_path)
    if not image_file.exists():
        return InvoiceData()

    with open(image_file, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    current_data = current_data or {}

    prompt = f"""
You are extracting invoice fields from an invoice image.

Important rules:
- Return JSON only.
- The document may be Arabic, English, mixed, or handwritten.
- Read directly from the image first. Use OCR text only as secondary support.
- Do not invent values.
- If a field is not readable enough, return null.
- Do not confuse phone numbers, customer numbers, reference numbers, order numbers, or row numbers with invoice_number.
- Do not translate vendor_name.
- If Arabic and English vendor names are both visible, prefer the Arabic official organization name.

General extraction priorities:
- Prefer printed header information over handwritten values when identifying vendor_name, invoice_number, invoice_date, and currency.
- Distinguish clearly between:
  1) official entity name
  2) address / district / street / industrial area / branch location
  3) customer information
- Use the image as the main source of truth. OCR text is only supporting context.

For vendor extraction:
- Look first in the top header/logo area.
- Prefer the official printed company or establishment name only.
- Do not include address text, district text, street text, branch text, industrial area text, delivery text, phone numbers, or customer text.
- If the top area contains both the company name and the address, return only the company name as vendor_name.
- Example:
  vendor_name = مؤسسة نجد الخليج للأدوات الصحية
  address/location text such as الشويخ الصناعية شارع الجوازات should NOT be included in vendor_name.

For invoice number extraction:
- Prefer short printed identifiers in the header area.
- The invoice number may appear near the top left, top center, or top right depending on the invoice layout.
- Prefer values near labels such as:
  invoice no, invoice number, inv no, رقم الفاتورة, فاتورة رقم
- If there is no explicit label, a short printed identifier in the header area may still be the invoice number if it is visually separated from the item table and amount columns.
- Do NOT use:
  - numbers from the item table
  - line numbers
  - quantity values
  - prices
  - totals
  - phone numbers
  - handwritten line values
- Reject any number that appears inside the item table or amount columns.

For invoice date extraction:
- Prefer values near:
  invoice date, date, تاريخ الفاتورة, التاريخ
- The invoice date is usually in the header area, not inside the item table or amount summary.
- The invoice date may be handwritten or printed.
- Accept common date formats such as:
  DD/MM/YYYY, D/M/YYYY, DD-MM-YYYY, YYYY/MM/DD
- If a handwritten date is visible near the header/date area, extract it exactly as a normal calendar date.
- Prefer 4-digit years when visible.
- Do not confuse invoice date with:
  due date, delivery date, ship date, print date, reference date, or row/table numbers.
- Do not use isolated numbers from the item table, quantity column, price column, or total boxes as invoice_date.
- If multiple date-like values are visible, prefer the one nearest to the header/date label.
- If the visible handwritten date looks like 12/02/2025, return 12/02/2025.
- If the date is not clearly readable, return null.

For total extraction:
- Total must be the final payable amount, not a line amount.
- Prefer the final payable amount shown in the bottom summary area of the invoice.
- Prefer values near labels such as:
  grand total, net total, final total, total due,
  الإجمالي النهائي, الإجمالي الكلي, المجموع النهائي, صافي الإجمالي, المبلغ الإجمالي, فقط
- A plain "total" may be valid only if it clearly refers to the final summary amount.
- Do NOT confuse total with:
  subtotal, tax, VAT, discount, unit price, rate, qty, quantity, item total, line total
  الضريبة، الخصم، السعر، الكمية، إجمالي السطر، الإجمالي الجزئي
- If several amounts are visible, choose the final bottom summary amount only.
- Ignore numbers inside the item table when selecting the final payable total.
- If the invoice uses separate fields or boxes for دينار and فلس, combine them into one Kuwaiti amount with exactly 3 decimal places.
- Example:
  دينار = 11
  فلس = 500
  total = 11.500
- Do not merge دينار and فلس as 115 or 115.0.
- For KWD / KD / د.ك, the فلس part is a 3-digit decimal part.
- If the amount is handwritten across separate boxes, read each box separately, then combine them correctly.
- If you see 11 in the دينار field and 500 in the فلس field, return total as 11.500 exactly.
- Do not infer total from summing line items if a final payable amount is visibly printed or handwritten in the summary area.
- If the final payable total is not clearly readable, return null for total.

For currency extraction:
- Prefer printed currency indicators such as KWD, KD, K.D., د.ك.
- If the invoice uses دينار / فلس amount fields, currency is KWD.

For items extraction:
- Extract line items only if they are visually readable enough.
- Do not invent missing unit_price or total_price.
- If some item lines are partially readable, return only what is visible.

Current OCR text:
{current_text[:4000]}

Current extracted data:
{json.dumps(current_data, ensure_ascii=False, indent=2)}

Return JSON exactly in this shape:
{{
  "vendor_name": null,
  "invoice_number": null,
  "invoice_date": null,
  "subtotal": null,
  "tax": null,
  "discount": null,
  "total": null,
  "currency": null,
  "items": [
    {{
      "description": null,
      "quantity": null,
      "unit_price": null,
      "total_price": null
    }}
  ]
}}

Final check before answering:
- Is vendor_name the official entity name only, without address/location text?
- Is invoice_number a short header identifier and not a table number, quantity, amount, or phone number?
- Is invoice_date visually readable near the date/header area?
- Is total the final payable amount at the bottom, not a line amount?
- If the invoice uses separate دينار and فلس fields, did you combine them correctly into a 3-decimal KWD amount?
- If unsure, return null instead of guessing.
""".strip()

    response = client.chat.completions.create(
        model=settings.model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            },
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