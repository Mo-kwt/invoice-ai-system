from pydantic import BaseModel, Field
from typing import List, Optional


class InvoiceItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None


class InvoiceData(BaseModel):
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    items: List[InvoiceItem] = Field(default_factory=list)


class ValidationResult(BaseModel):
    is_valid: bool
    needs_review: bool
    warnings: List[str] = Field(default_factory=list)