from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime, UTC
from app.db.database import Base


class InvoiceRecord(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, nullable=False)
    saved_path = Column(String, nullable=False)
    extracted_text = Column(Text, nullable=True)
    invoice_data_json = Column(Text, nullable=True)
    validation_result_json = Column(Text, nullable=True)
    status = Column(String, default="processed")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))