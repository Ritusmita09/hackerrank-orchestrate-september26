"""CSV transaction import — validated, idempotent, with row-level errors.

Rules (see spec §6 INGESTION):
* Required columns: date, direction, amount_minor (or a decimal ``amount``
  converted to minor units), category.
* Dates must parse as ISO ``YYYY-MM-DD``; direction must be credit|debit;
* amounts must be positive integers (or decimal amounts with at most two
  fractional digits); currency, if present, must match the user's home
  currency exactly (no silent conversion).
* Malformed rows are rejected individually — valid rows still import.
* The whole import is bounded by ``max_csv_rows``.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import User
from .audit_service import record_audit_event
from .transaction_service import TransactionServiceError, ingest_transaction

REQUIRED_COLUMNS = ("date", "direction", "category")
#: Either amount_minor (integer) or amount (decimal) must be present.
AMOUNT_COLUMNS = ("amount_minor", "amount")

VALID_DIRECTIONS = ("credit", "debit")
VALID_FLEXIBILITY = ("fixed", "reducible", "stoppable")


class CsvImportError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class RowError:
    row_number: int  # 1-based line number in the file (header = line 1)
    field: str
    message: str


@dataclass
class CsvImportResult:
    imported: int = 0
    rejected: int = 0
    duplicates: int = 0
    errors: list[RowError] = field(default_factory=list)


def _parse_date(raw: str, row_number: int) -> date:
    try:
        return date.fromisoformat(raw.strip())
    except ValueError as exc:
        raise CsvImportError("bad_date", f"row {row_number}: invalid date {raw!r}") from exc


def _parse_amount_minor(row: dict, row_number: int) -> int:
    """amount_minor (int) or amount (decimal, <= 2 fractional digits)."""
    raw_minor = (row.get("amount_minor") or "").strip()
    raw_amount = (row.get("amount") or "").strip()
    if raw_minor:
        try:
            value = int(raw_minor)
        except ValueError as exc:
            raise CsvImportError("bad_amount", f"row {row_number}: amount_minor must be an integer") from exc
    elif raw_amount:
        parts = raw_amount.split(".")
        if len(parts) > 2:
            raise CsvImportError("bad_amount", f"row {row_number}: invalid amount {raw_amount!r}")
        whole = parts[0].lstrip("+") or "0"
        frac = parts[1] if len(parts) == 2 else ""
        if len(frac) > 2 or not (whole.isdigit() and (not frac or frac.isdigit())):
            raise CsvImportError("bad_amount", f"row {row_number}: invalid amount {raw_amount!r}")
        value = int(whole) * 100 + int(frac.ljust(2, "0"))
    else:
        raise CsvImportError("missing_amount", f"row {row_number}: amount_minor or amount required")
    if value <= 0:
        raise CsvImportError("bad_amount", f"row {row_number}: amount must be positive")
    return value


def import_transactions_csv(
    db: Session,
    user: User,
    csv_text: str,
    *,
    source_id: str = "csv",
) -> CsvImportResult:
    """Validate and ingest a CSV of transactions. Commits nothing itself."""
    settings = get_settings()

    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        raise CsvImportError("empty_file", "the CSV file has no header row")

    columns = {c.strip().lower() for c in reader.fieldnames if c is not None}
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise CsvImportError("missing_columns", f"CSV is missing required column(s): {', '.join(missing)}")
    if not any(c in columns for c in AMOUNT_COLUMNS):
        raise CsvImportError(
            "missing_columns", "CSV needs an amount_minor or amount column"
        )

    result = CsvImportResult()
    for row_number, raw_row in enumerate(reader, start=2):  # header is line 1
        if row_number - 1 > settings.max_csv_rows:
            raise CsvImportError(
                "too_many_rows",
                f"CSV exceeds the maximum of {settings.max_csv_rows} rows",
            )
        row = {k.strip().lower(): (v or "").strip() for k, v in raw_row.items() if k is not None}

        try:
            row_date = _parse_date(row.get("date", ""), row_number)
            direction = row.get("direction", "").lower()
            if direction not in VALID_DIRECTIONS:
                raise CsvImportError(
                    "bad_direction",
                    f"row {row_number}: direction must be credit|debit, got {direction!r}",
                )
            currency = row.get("currency", "").upper() or user.home_currency
            if currency != user.home_currency:
                raise CsvImportError(
                    "currency_mismatch",
                    f"row {row_number}: currency {currency!r} does not match home currency {user.home_currency!r}",
                )
            amount_minor = _parse_amount_minor(row, row_number)
            flexibility = row.get("flexibility", "").lower() or "fixed"
            if flexibility not in VALID_FLEXIBILITY:
                raise CsvImportError(
                    "bad_flexibility",
                    f"row {row_number}: flexibility must be fixed|reducible|stoppable",
                )
            status = row.get("status", "").lower() or "settled"

            ingest_transaction(
                db, user,
                date_=row_date,
                direction=direction,
                amount_minor=amount_minor,
                category=row.get("category", "uncategorized") or "uncategorized",
                description=row.get("description", ""),
                status=status,
                flexibility=flexibility,
                source_type="csv",
                source_id=source_id,
            )
            result.imported += 1
        except CsvImportError as exc:
            result.rejected += 1
            result.errors.append(RowError(row_number=row_number, field="", message=exc.message))
        except TransactionServiceError as exc:
            result.rejected += 1
            if exc.code == "duplicate":
                result.duplicates += 1
            result.errors.append(
                RowError(row_number=row_number, field="", message=f"row {row_number}: {exc.message}")
            )

    record_audit_event(
        db, event_type="csv_imported",
        user_id=user.id,
        payload={
            "source_id": source_id,
            "imported": result.imported,
            "rejected": result.rejected,
            "duplicates": result.duplicates,
        },
    )
    db.flush()
    return result
