"""Goals store — user-stated saving/payoff/purchase intentions.

Backed by the ``financial_goals`` table.

A goal is what the user *wants*, not what they *owe*. It is deliberately not
an ``EventStatus``-bearing event: the deterministic engine never reads this
table, so a goal cannot become a scheduled payment, cannot move a forecast
balance, and cannot make an unaffordable purchase look affordable. When a goal
should influence a decision, it does so by informing the conversation — the
engine still decides using the ledger.

Money is integer minor units in the user's home currency, matching the rest of
the system (never floats).
"""
from __future__ import annotations

from datetime import date
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import FinancialGoal, User
from ..services.audit_service import record_audit_event
from .errors import MemoryServiceError
from .secrets import assert_storable

#: The kinds of goal the product supports (ARCHITECTURE.md §7 ``goals``).
GOAL_KINDS = ("save", "payoff", "purchase")

#: Free-text fields are bounded so memory cannot be used as bulk storage.
MAX_DESCRIPTION_CHARS = 2000


def _validate_kind(kind: str) -> str:
    cleaned = (kind or "").strip().lower()
    if cleaned not in GOAL_KINDS:
        raise MemoryServiceError(
            "invalid_goal",
            f"unknown goal kind {kind!r}; expected one of: {', '.join(GOAL_KINDS)}.",
        )
    return cleaned


def _validate_amount(target_amount_minor: int) -> int:
    try:
        amount = int(target_amount_minor)
    except (TypeError, ValueError):
        raise MemoryServiceError("invalid_goal", "target_amount_minor must be an integer.") from None
    if amount <= 0:
        raise MemoryServiceError(
            "invalid_goal", "target_amount_minor must be a positive amount in minor units."
        )
    return amount


def _validate_description(description: str) -> str:
    cleaned = (description or "").strip()
    if len(cleaned) > MAX_DESCRIPTION_CHARS:
        raise MemoryServiceError(
            "invalid_goal",
            f"goal descriptions are limited to {MAX_DESCRIPTION_CHARS} characters.",
        )
    return cleaned


def save_goal(
    db: Session,
    user: User,
    *,
    kind: str,
    target_amount_minor: int,
    target_date: date | None = None,
    priority: int = 0,
    description: str = "",
) -> FinancialGoal:
    """Record a goal. Explicit and audited.

    ``target_date`` is when the user hopes to reach the goal — an intention,
    not a commitment the engine will schedule.
    """
    cleaned_kind = _validate_kind(kind)
    amount = _validate_amount(target_amount_minor)
    text = _validate_description(description)
    # The description is user-authored free text: refuse credentials in it.
    assert_storable("description", text)

    goal = FinancialGoal(
        user_id=user.id,
        kind=cleaned_kind,
        target_amount_minor=amount,
        currency=user.home_currency,
        target_date=target_date,
        priority=int(priority or 0),
        description=text,
    )
    db.add(goal)
    record_audit_event(
        db,
        event_type="memory_goal_saved",
        user_id=user.id,
        payload={"goal_id": goal.id, "kind": cleaned_kind, "priority": goal.priority},
    )
    db.flush()
    return goal


def get_goal(db: Session, user: User, goal_id: str) -> FinancialGoal:
    """One goal, scoped to the user (another user's id is simply not found)."""
    goal = db.scalar(
        select(FinancialGoal)
        .where(FinancialGoal.id == goal_id)
        .where(FinancialGoal.user_id == user.id)
    )
    if goal is None:
        raise MemoryServiceError("not_found", f"goal {goal_id!r} does not exist")
    return goal


def list_goals(
    db: Session, user: User, *, include_achieved: bool = True
) -> Sequence[FinancialGoal]:
    """The user's goals, ordered deterministically (priority, then creation)."""
    query = select(FinancialGoal).where(FinancialGoal.user_id == user.id)
    if not include_achieved:
        query = query.where(FinancialGoal.achieved_at.is_(None))
    return db.scalars(
        query.order_by(
            FinancialGoal.priority.desc(),
            FinancialGoal.created_at,
            FinancialGoal.id,
        )
    ).all()


def update_goal(
    db: Session,
    user: User,
    goal_id: str,
    *,
    target_amount_minor: int | None = None,
    target_date: date | None = None,
    priority: int | None = None,
    description: str | None = None,
    achieved_at=None,
) -> FinancialGoal:
    """Update a goal in place (a goal is the user's own statement, not history)."""
    goal = get_goal(db, user, goal_id)
    changed: list[str] = []

    if target_amount_minor is not None:
        goal.target_amount_minor = _validate_amount(target_amount_minor)
        changed.append("target_amount_minor")
    if target_date is not None:
        goal.target_date = target_date
        changed.append("target_date")
    if priority is not None:
        goal.priority = int(priority)
        changed.append("priority")
    if description is not None:
        text = _validate_description(description)
        assert_storable("description", text)
        goal.description = text
        changed.append("description")
    if achieved_at is not None:
        goal.achieved_at = achieved_at
        changed.append("achieved_at")

    record_audit_event(
        db,
        event_type="memory_goal_updated",
        user_id=user.id,
        payload={"goal_id": goal.id, "fields": changed},
    )
    db.flush()
    return goal
