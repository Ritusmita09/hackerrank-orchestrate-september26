"""Recurring-pattern service — DB rows to engine patterns, and back.

Only patterns the user has confirmed (and that are still active) take part
in state reconstruction; unconfirmed detections are proposals and never
influence a decision.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.models import (
    Direction, Estimator, FixedPeriod, Flexibility, MonthlyPeriod, Money,
    RecurringPattern as EnginePattern,
)
from ..db.models import RecurringPattern, User


class PatternServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def row_to_engine_pattern(row: RecurringPattern) -> EnginePattern:
    """Translate one DB row into the engine's typed ``RecurringPattern``."""
    if row.period_type == "fixed":
        if row.period_days is None:
            raise PatternServiceError(
                "invalid_pattern",
                f"pattern {row.id} has period_type=fixed but no period_days",
            )
        period = FixedPeriod(days=row.period_days)
    elif row.period_type == "monthly":
        if not row.period_days_of_month:
            raise PatternServiceError(
                "invalid_pattern",
                f"pattern {row.id} has period_type=monthly but no anchor days",
            )
        period = MonthlyPeriod(days_of_month=tuple(row.period_days_of_month))
    else:
        raise PatternServiceError(
            "invalid_pattern", f"pattern {row.id} has unknown period_type {row.period_type!r}"
        )

    min_allowed = None
    if row.min_allowed_amount_minor is not None:
        min_allowed = Money.from_minor(row.min_allowed_amount_minor, row.currency)

    return EnginePattern(
        pattern_id=row.id,
        stream_key=row.stream_key,
        category=row.category,
        direction=Direction(row.direction),
        period=period,
        typical_amount=Money.from_minor(row.typical_amount_minor, row.currency),
        estimator=Estimator(row.estimator),
        last_seen=row.last_seen,
        evidence_event_ids=tuple(row.evidence_event_ids or ()),
        confidence=row.confidence,
        user_confirmed=row.user_confirmed,
        description=row.description or "",
        flexibility=Flexibility(row.flexibility or "fixed"),
        min_allowed_amount=min_allowed,
    )


def get_confirmed_patterns(db: Session, user: User) -> list[EnginePattern]:
    """Active, user-confirmed patterns for the user, as engine objects."""
    rows = db.scalars(
        select(RecurringPattern)
        .where(RecurringPattern.user_id == user.id)
        .where(RecurringPattern.active == True)
        .where(RecurringPattern.user_confirmed == True)
    ).all()
    return [row_to_engine_pattern(r) for r in rows]


def list_patterns(
    db: Session,
    user: User,
    *,
    confirmed: bool | None = None,
    include_inactive: bool = False,
) -> list[RecurringPattern]:
    """The user's patterns — proposals and confirmed — for the confirmation UI.

    ``confirmed=None`` returns both proposals and confirmed patterns;
    ``confirmed=False`` surfaces only unconfirmed proposals awaiting a user
    decision. Inactive (rejected) rows are hidden unless asked for.
    """
    stmt = select(RecurringPattern).where(RecurringPattern.user_id == user.id)
    if not include_inactive:
        stmt = stmt.where(RecurringPattern.active == True)
    if confirmed is not None:
        stmt = stmt.where(RecurringPattern.user_confirmed == confirmed)
    return list(
        db.scalars(
            stmt.order_by(
                RecurringPattern.created_at.desc(), RecurringPattern.id.desc()
            )
        ).all()
    )


def pattern_to_json(row: RecurringPattern) -> dict:
    """Serialize one pattern row for the API (JSON-safe, minor units)."""
    return {
        "id": row.id,
        "stream_key": row.stream_key,
        "category": row.category,
        "direction": row.direction,
        "period_type": row.period_type,
        "period_days": row.period_days,
        "period_days_of_month": row.period_days_of_month or [],
        "typical_amount_minor": row.typical_amount_minor,
        "estimator": row.estimator,
        "currency": row.currency,
        "last_seen": row.last_seen.isoformat(),
        "evidence_event_ids": row.evidence_event_ids or [],
        "confidence": row.confidence,
        "user_confirmed": row.user_confirmed,
        "active": row.active,
        "description": row.description or "",
        "flexibility": row.flexibility or "fixed",
        "min_allowed_amount_minor": row.min_allowed_amount_minor,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
