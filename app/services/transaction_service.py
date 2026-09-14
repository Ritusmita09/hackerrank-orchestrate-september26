"""Transactions service — append-oriented ingestion and retrieval."""
from __future__ import annotations

import hashlib
from datetime import date
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.models import Direction, EventStatus, Flexibility, Money
from ..db.models import Transaction, User
from .audit_service import record_audit_event


class TransactionServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _dedup_key(
    date_: date,
    direction: str,
    amount_minor: int,
    currency: str,
    description: str,
) -> str:
    """Canonicalize fields into a 64-char key to repel exact duplicate rows."""
    # Description spaces are condensed so "Acme Corp" and "Acme  Corp" match
    norm_desc = " ".join(description.strip().lower().split())
    segment = f"{date_.isoformat()}|{direction}|{amount_minor}|{currency}|{norm_desc}"
    return hashlib.sha256(segment.encode("utf-8")).hexdigest()


def ingest_transaction(
    db: Session,
    user: User,
    *,
    date_: date,
    direction: str,
    amount_minor: int,
    category: str,
    description: str = "",
    status: str = "settled",
    flexibility: str = "fixed",
    min_allowed_amount_minor: Optional[int] = None,
    source_type: str = "manual",
    source_id: str = "unknown",
    confidence: float = 1.0,
    confirmed_by_user: bool = True,
    amends_event_id: Optional[str] = None,
) -> Transaction:
    """Append one transaction to the ledger. Exact duplicates are rejected.

    The application must normalize the currency to the user's home currency
    *before* calling this. Any currency mismatch is rejected loudly.
    """
    if currency := user.home_currency:
        pass
    else:
        raise TransactionServiceError("missing_home_currency", "user has no home currency")

    try:
        Direction(direction)
        EventStatus(status)
        Flexibility(flexibility)
    except ValueError as exc:
        raise TransactionServiceError("invalid_enum", str(exc)) from exc

    key = _dedup_key(date_, direction, amount_minor, currency, description)
    existing = db.scalar(
        select(Transaction)
        .where(Transaction.user_id == user.id)
        .where(Transaction.dedup_key == key)
    )
    if existing is not None:
        raise TransactionServiceError(
            "duplicate",
            f"an identical transaction already exists (dedup {key[:8]})",
        )

    txn = Transaction(
        user_id=user.id,
        date=date_,
        direction=direction,
        amount_minor=amount_minor,
        currency=currency,
        category=category,
        description=description,
        status=status,
        flexibility=flexibility,
        min_allowed_amount_minor=min_allowed_amount_minor,
        source_type=source_type,
        source_id=source_id,
        confidence=confidence,
        confirmed_by_user=confirmed_by_user,
        amends_event_id=amends_event_id,
        dedup_key=key,
    )
    db.add(txn)
    record_audit_event(
        db, event_type="transaction_ingested",
        user_id=user.id,
        payload={"dedup_key": key, "source_type": source_type, "status": status},
    )
    db.flush()
    return txn


def get_transactions(
    db: Session,
    user_id: str,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    status: Optional[str] = None,
) -> Sequence[Transaction]:
    stmt = select(Transaction).where(Transaction.user_id == user_id)
    if start_date:
        stmt = stmt.where(Transaction.date >= start_date)
    if end_date:
        stmt = stmt.where(Transaction.date <= end_date)
    if status:
        stmt = stmt.where(Transaction.status == status)
    return db.scalars(stmt.order_by(Transaction.date.asc(), Transaction.id.asc())).all()


def row_to_financial_event(txn: Transaction) -> "FinancialEvent":
    """Translate one DB row into the engine's typed ``FinancialEvent``."""
    from ..core.models import (
        Direction, EventStatus, FinancialEvent, Flexibility, Money,
        Provenance, SourceType
    )

    amount = Money.from_minor(txn.amount_minor, txn.currency)
    min_allowed = None
    if txn.min_allowed_amount_minor is not None:
        min_allowed = Money.from_minor(txn.min_allowed_amount_minor, txn.currency)

    # Some older source_types might not map to the current enum; fallback.
    try:
        source_type = SourceType(txn.source_type)
    except ValueError:
        source_type = SourceType.MANUAL

    prov = Provenance(
        source_type=source_type,
        source_id=txn.source_id or "",
        confidence=txn.confidence,
        confirmed_by_user=txn.confirmed_by_user,
    )

    return FinancialEvent(
        event_id=txn.id,
        date=txn.date,
        direction=Direction(txn.direction),
        amount=amount,
        category=txn.category,
        description=txn.description,
        status=EventStatus(txn.status),
        flexibility=Flexibility(txn.flexibility),
        min_allowed_amount=min_allowed,
        provenance=prov,
        linked_event_id=txn.linked_event_id,
        amends_event_id=txn.amends_event_id,
    )
