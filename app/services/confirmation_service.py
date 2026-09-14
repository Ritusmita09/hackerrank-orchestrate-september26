"""Confirmation flows — where inferred facts become ledger truth.

ARCHITECTURE.md §5.2: inferred facts are always drafts until confirmed. This
module is the single gate through which the two existing kinds of inferred
fact cross that line:

* **Extraction drafts** (``DocumentExtraction`` rows, status ``draft``).
  ``confirm_extraction`` ingests exactly one transaction derived from the
  draft, with provenance ``source_type=document, source_id=<document_id>``
  (so deleting the document can cascade), then marks the extraction
  ``confirmed``. ``reject_extraction`` marks it ``rejected`` — a rejected
  draft is re-extracted on the next ``extract_document`` run (the cache only
  reuses ``draft`` and ``confirmed`` rows).
* **Pattern proposals** (``RecurringPattern`` rows, ``user_confirmed=False``).
  Only the user makes a proposal authoritative; ``confirm_pattern`` sets the
  flag, ``reject_pattern`` retires the pattern (``active=False``) so it never
  influences a forecast again.

Boundaries enforced here rather than requested in a prompt:

* **Confirm means ingest once.** A non-draft extraction cannot be confirmed
  again — there is no path by which one confirmation produces two ledger
  events. The ledger's dedup key is the backstop if two different drafts
  carry identical values.
* **No currency conversion.** A draft whose currency is not the user's home
  currency is refused: converting would invent an exchange rate, and money is
  never invented.
* **Amounts come from the validated draft only.** The user may correct
  direction, category, description, and the event date at confirmation; the
  amount is the schema-validated total, nothing re-derived.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional, Tuple

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import DocumentExtraction, RecurringPattern, Transaction, User
from .audit_service import record_audit_event
from .extraction_service import ExtractionDraft, get_extraction
from .transaction_service import TransactionServiceError, ingest_transaction


class ConfirmationServiceError(Exception):
    """Domain error carrying a machine-readable code (service convention)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: Default cash-flow direction per document type. A payslip is money in;
#: bills, invoices, and receipts are money out. Statements and free-form
#: documents carry no default — the caller must state one.
_DIRECTION_BY_TYPE = {
    "payslip": "credit",
    "bill": "debit",
    "invoice": "debit",
    "receipt": "debit",
}

#: Which draft date is the *event* date, per document type: a bill/invoice
#: hits the ledger on its due date, a payslip/receipt on its issue date.
_DATE_FIELD_BY_TYPE = {
    "payslip": "issue_date",
    "receipt": "issue_date",
    "bill": "due_date",
    "invoice": "due_date",
}

_VALID_DIRECTIONS = ("credit", "debit")


def _draft_confidence(draft: ExtractionDraft) -> float:
    """The confidence carried onto the ingested event.

    The amount is the number that matters, so its own reported confidence is
    preferred; otherwise the mean of the reported fields; a draft with no
    confidence map at all is treated as fully confident.
    """
    confidences = draft.field_confidence or {}
    if "total_amount_minor" in confidences:
        return float(confidences["total_amount_minor"])
    if confidences:
        return sum(float(c) for c in confidences.values()) / len(confidences)
    return 1.0


# ---------------------------------------------------------------------------
# Extraction confirmation / rejection
# ---------------------------------------------------------------------------

def confirm_extraction(
    db: Session,
    user: User,
    *,
    extraction_id: str,
    as_of: date,
    direction: Optional[str] = None,
    category: Optional[str] = None,
    description: Optional[str] = None,
    event_date: Optional[date] = None,
) -> Tuple[DocumentExtraction, Transaction]:
    """Confirm a draft extraction and ingest it as exactly one transaction.

    Only ``draft``-status extractions can be confirmed. The caller may correct
    the mapping to the ledger (direction, category, description, event date);
    the amount and currency always come from the validated draft.

    Returns ``(extraction, transaction)``; raises ``ConfirmationServiceError``
    on any refusal (never a partial ingest).
    """
    try:
        extraction = get_extraction(db, user, extraction_id)
    except Exception as exc:
        # get_extraction raises ExtractionServiceError("extraction_not_found")
        code = getattr(exc, "code", "extraction_not_found")
        raise ConfirmationServiceError(code, str(exc)) from None

    if extraction.status != "draft":
        raise ConfirmationServiceError(
            "invalid_status",
            f"extraction {extraction_id!r} is {extraction.status!r}; "
            f"only drafts can be confirmed",
        )

    try:
        draft = ExtractionDraft.model_validate(extraction.draft or {})
    except ValidationError:
        raise ConfirmationServiceError(
            "invalid_draft",
            "the stored draft no longer validates against the extraction schema",
        ) from None

    # Direction: caller override, else the type default, else a refusal.
    resolved_direction = (direction or "").strip().lower() or _DIRECTION_BY_TYPE.get(
        draft.document_type, ""
    )
    if not resolved_direction:
        raise ConfirmationServiceError(
            "direction_required",
            f"document_type {draft.document_type!r} has no default direction; "
            f"pass direction='credit' or 'debit'",
        )
    if resolved_direction not in _VALID_DIRECTIONS:
        raise ConfirmationServiceError(
            "invalid_direction",
            f"direction must be credit or debit, got {resolved_direction!r}",
        )

    # No exchange rates, ever: a foreign-currency draft must be refused
    # rather than silently converted.
    if draft.currency != user.home_currency:
        raise ConfirmationServiceError(
            "currency_mismatch",
            f"the draft is in {draft.currency} but the user's home currency is "
            f"{user.home_currency}; currency is never converted on ingest",
        )

    # Event date: caller override, else the type's natural date, else any
    # date the draft has, else a refusal (never guess a date).
    resolved_date = event_date
    if resolved_date is None:
        field = _DATE_FIELD_BY_TYPE.get(draft.document_type)
        resolved_date = getattr(draft, field, None) if field else None
    if resolved_date is None:
        resolved_date = draft.due_date or draft.issue_date
    if resolved_date is None:
        raise ConfirmationServiceError(
            "date_required",
            "the draft has no usable date; pass event_date explicitly",
        )

    # A confirmed future flow is scheduled (counted in forecasts); a past
    # one is settled history.
    status = "settled" if resolved_date <= as_of else "scheduled"

    resolved_category = (category or "").strip() or draft.document_type
    if description is not None:
        resolved_description = description
    else:
        resolved_description = (
            f"{draft.issuer} {draft.document_type}".strip() or draft.document_type
        )

    try:
        txn = ingest_transaction(
            db,
            user,
            date_=resolved_date,
            direction=resolved_direction,
            amount_minor=draft.total_amount_minor,
            category=resolved_category,
            description=resolved_description,
            status=status,
            source_type="document",
            source_id=extraction.document_id,
            confidence=_draft_confidence(draft),
            confirmed_by_user=True,
        )
    except TransactionServiceError as exc:
        code = "duplicate_event" if exc.code == "duplicate" else exc.code
        raise ConfirmationServiceError(code, exc.message) from None

    extraction.status = "confirmed"
    extraction.confirmed_at = _utcnow()

    record_audit_event(
        db,
        event_type="extraction_confirmed",
        user_id=user.id,
        payload={
            "extraction_id": extraction.id,
            "document_id": extraction.document_id,
            "transaction_id": txn.id,
            "direction": resolved_direction,
            "event_date": resolved_date.isoformat(),
            "status": status,
        },
    )
    db.flush()
    return extraction, txn


def reject_extraction(
    db: Session, user: User, *, extraction_id: str
) -> DocumentExtraction:
    """Reject a draft extraction: the model's read of the document was wrong.

    Idempotent for already-rejected drafts. A confirmed extraction cannot be
    rejected here — its transaction is already in the ledger; removing ledger
    events is a correction flow, not a confirmation flow.
    """
    try:
        extraction = get_extraction(db, user, extraction_id)
    except Exception as exc:
        code = getattr(exc, "code", "extraction_not_found")
        raise ConfirmationServiceError(code, str(exc)) from None

    if extraction.status == "rejected":
        return extraction
    if extraction.status != "draft":
        raise ConfirmationServiceError(
            "invalid_status",
            f"extraction {extraction_id!r} is {extraction.status!r}; only drafts "
            f"can be rejected (a confirmed extraction is already ingested)",
        )

    extraction.status = "rejected"
    record_audit_event(
        db,
        event_type="extraction_rejected",
        user_id=user.id,
        payload={"extraction_id": extraction.id, "document_id": extraction.document_id},
    )
    db.flush()
    return extraction


# ---------------------------------------------------------------------------
# Pattern confirmation / rejection
# ---------------------------------------------------------------------------

def _get_pattern_row(db: Session, user: User, pattern_id: str) -> RecurringPattern:
    row = db.scalar(
        select(RecurringPattern)
        .where(RecurringPattern.id == pattern_id)
        .where(RecurringPattern.user_id == user.id)
    )
    if row is None:
        raise ConfirmationServiceError(
            "pattern_not_found", f"pattern {pattern_id!r} does not exist"
        )
    return row


def confirm_pattern(db: Session, user: User, *, pattern_id: str) -> RecurringPattern:
    """Make a detected recurring-stream proposal authoritative.

    Until this happens the pattern never takes part in state reconstruction
    (``pattern_service.get_confirmed_patterns`` filters on the flag), so an
    unconfirmed detection can never influence a forecast or a decision.
    Idempotent: confirming an already-confirmed pattern changes nothing.
    """
    row = _get_pattern_row(db, user, pattern_id)

    if not row.active:
        raise ConfirmationServiceError(
            "pattern_inactive",
            f"pattern {pattern_id!r} is inactive and cannot be confirmed "
            f"(it may have been rejected earlier)",
        )
    if row.user_confirmed:
        return row

    row.user_confirmed = True
    record_audit_event(
        db,
        event_type="pattern_confirmed",
        user_id=user.id,
        payload={
            "pattern_id": row.id,
            "stream_key": row.stream_key,
            "confidence": row.confidence,
        },
    )
    db.flush()
    return row


def reject_pattern(db: Session, user: User, *, pattern_id: str) -> RecurringPattern:
    """Retire a pattern proposal (or a confirmed pattern the user disagrees with).

    Rejection deactivates the row: it stops participating in state
    reconstruction immediately, whether or not it had been confirmed. This
    does not remove past transactions — it only stops future occurrences
    being generated from the pattern. Idempotent.
    """
    row = _get_pattern_row(db, user, pattern_id)

    if not row.active:
        return row

    row.active = False
    record_audit_event(
        db,
        event_type="pattern_rejected",
        user_id=user.id,
        payload={"pattern_id": row.id, "stream_key": row.stream_key},
    )
    db.flush()
    return row
