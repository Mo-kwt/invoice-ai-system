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
- Do not confuse phone numbers, customer numbers, reference numbers, or order numbers with invoice_number.
- Do not translate vendor_name.
- If Arabic and English vendor names are both visible, prefer the Arabic official organization name.
- Vendor name is usually in the top header area of the invoice.
- Invoice number and invoice date are usually in the upper or middle header area.
- Total must be the final payable amount, not a line amount.

For total extraction:
- Prefer the final payable amount shown in the bottom summary area of the invoice.
- Prefer values near labels such as:
  grand total, net total, final total, total due,
  الإجمالي النهائي, الإجمالي الكلي, المجموع النهائي, صافي الإجمالي, المبلغ الإجمالي, فقط
- A plain "total" may be valid only if it clearly refers to the final summary amount.
- Do NOT confuse total with:
  subtotal, tax, VAT, discount, unit price, rate, qty, quantity, item total, line total
  الضريبة، الخصم، السعر، الكمية، إجمالي السطر، الإجمالي الجزئي
- If several amounts are visible, choose the final bottom summary amount only.
- If the image does not clearly show a final payable total, return null for total.

For vendor extraction:
- Look first at the top header/logo area.
- Prefer the printed company or establishment name, not address text, street text, or customer text.

For invoice number extraction:
- Prefer values near labels like:
  invoice no, invoice number, inv no, رقم الفاتورة, فاتورة رقم
- Reject phone numbers and money amounts.

For invoice date extraction:
- Prefer values near:
  invoice date, date, تاريخ الفاتورة, التاريخ
- Do not confuse with due date, delivery date, ship date, or print date.

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
- Is vendor_name actually visible in the top header?
- Is invoice_number visually readable and not a phone/reference number?
- Is invoice_date visually readable?
- Is total the final payable amount at the bottom, not subtotal or line amount?
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