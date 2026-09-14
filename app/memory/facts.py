"""Facts store — dated, user-stated statements with validity windows.

Backed by the ``financial_facts`` table ("contract ends 2027-03", "rent moves
to the 3rd in April", "I'm between jobs until June").

Each fact carries ``valid_from`` / ``valid_to``. Reads take an ``as_of`` date
and exclude facts outside their window, so an expired fact stops colouring the
conversation on its own — no cleanup job required (ARCHITECTURE.md §10).

Facts are narration the user supplied, not financial records. A fact like
"I earn 4000/month" does not become income in a forecast: income enters the
engine only through the ledger and confirmed recurring patterns.
"""
from __future__ import annotations

from datetime import date
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import FinancialFact, User
from ..services.audit_service import record_audit_event
from .errors import MemoryServiceError
from .secrets import assert_storable

MAX_FACT_CHARS = 4000

#: Where a fact came from. ``manual`` is a user statement; later phases add
#: ``document`` / ``message`` once confirmation flows exist.
FACT_SOURCES = ("manual", "user_stated", "conversation")


def _validate_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        raise MemoryServiceError("invalid_fact", "fact text is required.")
    if len(cleaned) > MAX_FACT_CHARS:
        raise MemoryServiceError(
            "invalid_fact", f"fact text is limited to {MAX_FACT_CHARS} characters."
        )
    return cleaned


def _validate_window(valid_from: date | None, valid_to: date | None) -> None:
    if valid_from is not None and valid_to is not None and valid_to < valid_from:
        raise MemoryServiceError(
            "invalid_fact", "valid_to cannot be earlier than valid_from."
        )


def save_fact(
    db: Session,
    user: User,
    *,
    text: str,
    source: str = "manual",
    valid_from: date | None = None,
    valid_to: date | None = None,
) -> FinancialFact:
    """Store one fact. Explicit and audited."""
    cleaned = _validate_text(text)
    assert_storable("fact", cleaned)
    _validate_window(valid_from, valid_to)

    cleaned_source = (source or "manual").strip().lower()
    if cleaned_source not in FACT_SOURCES:
        raise MemoryServiceError(
            "invalid_fact",
            f"unknown fact source {source!r}; expected one of: {', '.join(FACT_SOURCES)}.",
        )

    fact = FinancialFact(
        user_id=user.id,
        text=cleaned,
        source=cleaned_source,
        valid_from=valid_from,
        valid_to=valid_to,
    )
    db.add(fact)
    record_audit_event(
        db,
        event_type="memory_fact_saved",
        user_id=user.id,
        payload={"fact_id": fact.id, "source": cleaned_source},
    )
    db.flush()
    return fact


def get_fact(db: Session, user: User, fact_id: str) -> FinancialFact:
    """One fact, scoped to the user."""
    fact = db.scalar(
        select(FinancialFact)
        .where(FinancialFact.id == fact_id)
        .where(FinancialFact.user_id == user.id)
    )
    if fact is None:
        raise MemoryServiceError("not_found", f"fact {fact_id!r} does not exist")
    return fact


def _in_window(as_of: date | None):
    """SQL predicate: the fact is inside its validity window at ``as_of``."""
    if as_of is None:
        return None
    return (
        (FinancialFact.valid_from.is_(None)) | (FinancialFact.valid_from <= as_of),
        (FinancialFact.valid_to.is_(None)) | (FinancialFact.valid_to >= as_of),
    )


def list_active_facts(
    db: Session, user: User, *, as_of: date | None = None
) -> Sequence[FinancialFact]:
    """Facts valid at ``as_of`` (all facts when ``as_of`` is omitted).

    Ordered by creation then id so repeated reads are byte-identical.
    """
    query = select(FinancialFact).where(FinancialFact.user_id == user.id)
    for predicate in _in_window(as_of) or ():
        query = query.where(predicate)
    return db.scalars(
        query.order_by(FinancialFact.created_at, FinancialFact.id)
    ).all()


def search_facts(
    db: Session,
    user: User,
    *,
    query_text: str = "",
    as_of: date | None = None,
) -> Sequence[FinancialFact]:
    """Substring search over the user's facts, deterministically ordered.

    Deliberately a plain, predictable match — no ranking heuristics that could
    make retrieval non-reproducible.
    """
    stmt = select(FinancialFact).where(FinancialFact.user_id == user.id)
    cleaned = (query_text or "").strip()
    if cleaned:
        stmt = stmt.where(FinancialFact.text.ilike(f"%{cleaned}%"))
    for predicate in _in_window(as_of) or ():
        stmt = stmt.where(predicate)
    return db.scalars(stmt.order_by(FinancialFact.created_at, FinancialFact.id)).all()


def close_fact(db: Session, user: User, fact_id: str, *, valid_to: date) -> FinancialFact:
    """End a fact's validity window, so it stops influencing later turns."""
    fact = get_fact(db, user, fact_id)
    _validate_window(fact.valid_from, valid_to)
    fact.valid_to = valid_to
    record_audit_event(
        db, event_type="memory_fact_closed", user_id=user.id, payload={"fact_id": fact.id}
    )
    db.flush()
    return fact
