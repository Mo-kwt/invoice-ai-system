import json
from openai import OpenAI
from app.config import settings
from app.schemas.invoice import InvoiceData

client = OpenAI(api_key=settings.openai_api_key)


def extract_invoice_data_from_text(text: str) -> InvoiceData:
    if not text or not text.strip():
        return InvoiceData()

    prompt = f"""
You are an expert document data extraction assistant.

Extract the following fields from the provided document text and return JSON only.

Required JSON structure:
{{
  "vendor_name": string or null,
  "invoice_number": string or null,
  "invoice_date": string or null,
  "subtotal": number or null,
  "tax": number or null,
  "discount": number or null,
  "total": number or null,
  "currency": string or null,
  "items": [
    {{
      "description": string or null,
      "quantity": number or null,
      "unit_price": number or null,
      "total_price": number or null
    }}
  ]
}}

Rules:
- Return valid JSON only.
- Do not add markdown.
- If a field is missing, use null.
- If the document is not an invoice, extract the closest matching structured fields where possible.
- Preserve numbers as numbers, not strings.

Document text:
{text[:12000]}
"""

    response = client.chat.completions.create(
        model=settings.model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You extract structured data from business documents."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    return InvoiceData(**data)