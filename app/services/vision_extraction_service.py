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
- Prefer fields that are visually visible in the image even if OCR text is weak.
- Do not invent values.
- If invoice number or invoice date are not readable enough, return null.
- Do not confuse phone numbers with invoice numbers.
- Do not translate vendor_name.
- If Arabic vendor name is visible, prefer Arabic.
- If total is clearly visible near Total / Amount / المبلغ / الإجمالي, extract it.
- Items may be empty if not readable.

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