from app.schemas.invoice import InvoiceData, ValidationResult


def validate_invoice_data(data: InvoiceData) -> ValidationResult:
    warnings = []

    if not data.vendor_name:
        warnings.append("Vendor name is missing.")

    if not data.invoice_number:
        warnings.append("Invoice number is missing.")

    if not data.invoice_date:
        warnings.append("Invoice date is missing.")

    if data.total is None:
        warnings.append("Total amount is missing.")

    if data.currency is None:
        warnings.append("Currency is missing.")

    expected_total = None
    if data.subtotal is not None:
        expected_total = data.subtotal + (data.tax or 0) - (data.discount or 0)

    if expected_total is not None and data.total is not None:
        if round(expected_total, 2) != round(data.total, 2):
            warnings.append(
                f"Total mismatch: expected {round(expected_total, 2)} but got {round(data.total, 2)}."
            )

    if data.items:
        for idx, item in enumerate(data.items, start=1):
            if item.quantity is not None and item.unit_price is not None and item.total_price is not None:
                expected_item_total = item.quantity * item.unit_price
                if round(expected_item_total, 2) != round(item.total_price, 2):
                    warnings.append(
                        f"Item {idx} total mismatch: expected {round(expected_item_total, 2)} "
                        f"but got {round(item.total_price, 2)}."
                    )

    is_valid = len(warnings) == 0
    needs_review = not is_valid

    return ValidationResult(
        is_valid=is_valid,
        needs_review=needs_review,
        warnings=warnings
    )