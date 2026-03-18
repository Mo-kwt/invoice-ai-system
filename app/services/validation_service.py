from app.schemas.invoice import InvoiceData, ValidationResult


def validate_invoice_data(invoice: InvoiceData) -> ValidationResult:
    warnings = []

    # 1) التحقق من الحقول الأساسية
    if not invoice.vendor_name:
        warnings.append("Vendor name is missing.")

    if not invoice.invoice_number:
        warnings.append("Invoice number is missing.")

    if not invoice.invoice_date:
        warnings.append("Invoice date is missing.")

    if invoice.total is None:
        warnings.append("Total amount is missing.")

    if not invoice.currency:
        warnings.append("Currency is missing.")

    # 2) فحص منطقي بسيط للمبالغ
    subtotal = invoice.subtotal
    tax = invoice.tax
    discount = invoice.discount
    total = invoice.total

    if subtotal is not None and tax is not None and total is not None:
        expected_total = subtotal + tax
        if discount is not None:
            expected_total -= discount

        if abs(expected_total - total) > 0.01:
            warnings.append(
                f"Total mismatch: expected {expected_total:.3f}, got {total:.3f}."
            )

    # 3) تقييم الذكاء العام للنتيجة
    critical_missing = 0

    if not invoice.invoice_number:
        critical_missing += 1

    if not invoice.invoice_date:
        critical_missing += 1

    if invoice.total is None:
        critical_missing += 1

    # 4) هل النتيجة صالحة؟
    # نعتبرها صالحة فقط إذا لم توجد مشاكل حرجة
    is_valid = critical_missing == 0

    # 5) هل تحتاج مراجعة؟
    # أي نقص أو تحذير مهم يعني needs_review
    needs_review = len(warnings) > 0

    return ValidationResult(
        is_valid=is_valid,
        needs_review=needs_review,
        warnings=warnings
    )