from app.schemas.invoice import InvoiceData, ValidationResult


def _has_meaningful_text(value: str | None) -> bool:
    if value is None:
        return False
    return bool(str(value).strip())


def validate_invoice_data(invoice: InvoiceData) -> ValidationResult:
    warnings = []
    review_reasons = []
    missing_fields = []
    score = 100.0

    # نجمع أسباب المراجعة الحرجة بشكل منفصل
    critical_review_triggers = []

    # 1) التحقق من الحقول الأساسية
    if not _has_meaningful_text(invoice.vendor_name):
        warnings.append("Vendor name is missing.")
        review_reasons.append("Missing vendor name")
        missing_fields.append("vendor_name")
        score -= 15

    if not _has_meaningful_text(invoice.invoice_number):
        warnings.append("Invoice number is missing.")
        review_reasons.append("Missing invoice number")
        missing_fields.append("invoice_number")
        critical_review_triggers.append("missing_invoice_number")
        score -= 20

    if not _has_meaningful_text(invoice.invoice_date):
        warnings.append("Invoice date is missing.")
        review_reasons.append("Missing invoice date")
        missing_fields.append("invoice_date")
        critical_review_triggers.append("missing_invoice_date")
        score -= 20

    if invoice.total is None:
        warnings.append("Total amount is missing.")
        review_reasons.append("Missing total amount")
        missing_fields.append("total")
        critical_review_triggers.append("missing_total")
        score -= 25

    if not _has_meaningful_text(invoice.currency):
        warnings.append("Currency is missing.")
        review_reasons.append("Missing currency")
        missing_fields.append("currency")
        score -= 10

    # 2) فحص منطقي للمبالغ
    subtotal = invoice.subtotal
    tax = invoice.tax
    discount = invoice.discount
    total = invoice.total

    if subtotal is not None and subtotal < 0:
        warnings.append("Subtotal is negative.")
        review_reasons.append("Negative subtotal")
        critical_review_triggers.append("negative_subtotal")
        score -= 10

    if tax is not None and tax < 0:
        warnings.append("Tax is negative.")
        review_reasons.append("Negative tax")
        critical_review_triggers.append("negative_tax")
        score -= 10

    if discount is not None and discount < 0:
        warnings.append("Discount is negative.")
        review_reasons.append("Negative discount")
        critical_review_triggers.append("negative_discount")
        score -= 8

    if total is not None and total < 0:
        warnings.append("Total amount is negative.")
        review_reasons.append("Negative total amount")
        critical_review_triggers.append("negative_total")
        score -= 20

    if subtotal is not None and tax is not None and total is not None:
        expected_total = subtotal + tax
        if discount is not None:
            expected_total -= discount

        difference = abs(expected_total - total)
        if difference > 0.01:
            warnings.append(
                f"Total mismatch: expected {expected_total:.3f}, got {total:.3f}."
            )
            review_reasons.append("Total does not match subtotal + tax - discount")
            critical_review_triggers.append("total_mismatch")
            score -= 20

    # 3) فحص عناصر الفاتورة
    # غياب items مهم، لكنه ليس دائمًا سببًا كافيًا وحده لإجبار المراجعة
    if not invoice.items:
        warnings.append("No line items were extracted.")
        review_reasons.append("No invoice items extracted")
        score -= 8
    else:
        bad_items = 0
        for item in invoice.items:
            has_desc = _has_meaningful_text(item.description)
            has_any_number = (
                item.quantity is not None
                or item.unit_price is not None
                or item.total_price is not None
            )
            if not has_desc and not has_any_number:
                bad_items += 1

        if bad_items > 0:
            warnings.append(f"{bad_items} extracted item(s) look incomplete.")
            review_reasons.append("Some invoice items look incomplete")
            score -= min(12, bad_items * 3)

            # نعدها حرجة فقط إذا كانت كل العناصر تقريبًا رديئة
            if invoice.items and bad_items >= len(invoice.items):
                critical_review_triggers.append("all_items_incomplete")

    # 4) قواعد إضافية مفيدة عمليًا
    if _has_meaningful_text(invoice.vendor_name) and len(invoice.vendor_name.strip()) < 3:
        warnings.append("Vendor name looks too short.")
        review_reasons.append("Vendor name looks suspiciously short")
        score -= 10

    if _has_meaningful_text(invoice.invoice_number) and len(invoice.invoice_number.strip()) < 2:
        warnings.append("Invoice number looks too short.")
        review_reasons.append("Invoice number looks suspiciously short")
        critical_review_triggers.append("invoice_number_too_short")
        score -= 10

    # 5) منع الدرجة من النزول أقل من صفر
    score = max(0.0, min(100.0, score))

    # 6) منطق الصلاحية
    critical_missing = 0
    if not _has_meaningful_text(invoice.invoice_number):
        critical_missing += 1
    if not _has_meaningful_text(invoice.invoice_date):
        critical_missing += 1
    if invoice.total is None:
        critical_missing += 1

    is_valid = critical_missing == 0

    # 7) منطق المراجعة الجديد
    # تحتاج مراجعة إذا:
    # - يوجد trigger حرج
    # - أو الدرجة منخفضة
    # - أو هناك أكثر من حقلين ناقصين عمومًا
    # أما warnings البسيطة وحدها فلا تفرض مراجعة دائمًا
    needs_review = (
        len(critical_review_triggers) > 0
        or score < 85
        or len(missing_fields) >= 3
    )

    return ValidationResult(
        is_valid=is_valid,
        needs_review=needs_review,
        warnings=warnings,
        confidence_score=round(score, 2),
        review_reasons=review_reasons,
        missing_fields=missing_fields,
    )