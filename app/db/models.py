from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base

from sqlalchemy import Column, Boolean

used_fallback = Column(Boolean, default=False)


# =========================
# NEW STRUCTURE
# =========================
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String)
    file_path = Column(String)
    mime_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    runs = relationship("ProcessingRun", back_populates="document", cascade="all, delete")


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))

    pipeline_version = Column(String, default="v1")
    model_name = Column(String)
    status = Column(String, default="processing")

    used_fallback = Column(Boolean, default=False)
    fallback_reasons_json = Column(Text)

    classification_json = Column(Text)
    validation_json = Column(Text)
    final_output_json = Column(Text)

    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)

    document = relationship("Document", back_populates="runs")
    steps = relationship("ProcessingStep", back_populates="run", cascade="all, delete")
    reviews = relationship("ReviewAction", back_populates="run", cascade="all, delete")


class ProcessingStep(Base):
    __tablename__ = "processing_steps"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("processing_runs.id"))

    step_name = Column(String)
    status = Column(String)

    input_snapshot_json = Column(Text)
    output_snapshot_json = Column(Text)
    error_message = Column(Text)

    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)

    run = relationship("ProcessingRun", back_populates="steps")


class ReviewAction(Base):
    __tablename__ = "review_actions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    run_id = Column(Integer, ForeignKey("processing_runs.id"))

    action_type = Column(String)
    actor = Column(String)

    before_json = Column(Text)
    after_json = Column(Text)
    note = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("ProcessingRun", back_populates="reviews")


# =========================
# TEMPORARY BACKWARD COMPATIBILITY
# =========================
from sqlalchemy import Boolean  # ✅ تأكد من إضافته بالأعلى

class InvoiceRecord(Base):
    __tablename__ = "invoice_records"

    id = Column(Integer, primary_key=True, index=True)

    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    run_id = Column(Integer, ForeignKey("processing_runs.id"), nullable=True)

    original_filename = Column(String)
    saved_path = Column(String)
    extracted_text = Column(Text)

    invoice_data_json = Column(Text)
    validation_result_json = Column(Text)

    status = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)

    used_fallback = Column(Boolean, default=False)  # ✅ هذا هو التعديل