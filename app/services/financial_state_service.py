"""Financial-state reconstruction — persistence rows to engine ``FinancialState``.

This is the single place the app's stored history becomes the normalized,
deterministic ``FinancialState`` the engine consumes. It delegates conflict
resolution, amendments, and stream synthesis to the engine's ``build_state``
(Phase 1, locked).
"""
from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.models import FinancialState
from ..core.state import BuildContext, build_state
from ..db.models import User
from .pattern_service import get_confirmed_patterns
from .transaction_service import row_to_financial_event
from .user_service import get_latest_profile, profile_to_engine_profile


def get_financial_state(db: Session, user: User, as_of: date) -> FinancialState:
    """Load the user's stored history and build the normalized state.

    Inputs: latest profile revision, every transaction row (confirmed and
    historical — the engine's conflict resolution decides what counts), and
    the active user-confirmed recurring patterns.
    """
    db_profile = get_latest_profile(db, user.id)
    engine_profile = profile_to_engine_profile(db_profile, user.id, as_of)

    from ..db.models import Transaction

    txs = db.scalars(
        select(Transaction).where(Transaction.user_id == user.id)
    ).all()
    events = [row_to_financial_event(t) for t in txs]
    patterns = get_confirmed_patterns(db, user)

    ctx = BuildContext(
        user_id=user.id,
        as_of=as_of,
        profile=engine_profile,
        events=tuple(events),
        patterns=tuple(patterns),
    )
    return build_state(ctx)


def state_to_json(state: FinancialState) -> dict:
    """Serialize the reconstructed state for the API (JSON-safe, minor units).

    Streams are summarized rather than dumped in full: the endpoint's job is
    to show the user what the engine sees, not to re-expose raw rows.
    """
    def stream_json(streams, is_expense: bool):
        out = []
        for s in streams:
            item = {
                "stream_key": s.stream_key,
                "category": s.category,
                "description": s.description,
                "typical_amount_minor": s.typical_amount.amount_minor if s.typical_amount else None,
                "history_events": len(s.history),
                "future_confirmed_events": len(s.future_confirmed),
            }
            if s.pattern is not None:
                period = s.pattern.period
                item["pattern"] = {
                    "confidence": s.pattern.confidence,
                    "user_confirmed": s.pattern.user_confirmed,
                    "estimator": s.pattern.estimator.value,
                    "flexibility": s.pattern.flexibility.value,
                }
            if is_expense:
                item["flexibility"] = s.flexibility.value
                item["is_protected"] = s.is_protected
                item["can_reduce"] = s.can_reduce
                item["can_stop"] = s.can_stop
                if s.min_allowed_amount is not None:
                    item["min_allowed_amount_minor"] = s.min_allowed_amount.amount_minor
            out.append(item)
        return out

    return {
        "user_id": state.user_id,
        "as_of": state.as_of.isoformat(),
        "home_currency": state.home_currency,
        "current_balance_minor": state.current_balance.amount_minor,
        "minimum_balance_minor": state.minimum_balance.amount_minor,
        "profile": {
            "current_balance_minor": state.profile.current_available_balance.amount_minor,
            "minimum_balance_minor": state.profile.minimum_balance_to_keep.amount_minor,
            "protected_categories": sorted(state.profile.protected_categories),
            "reducible_categories": sorted(state.profile.reducible_categories),
            "stoppable_categories": sorted(state.profile.stoppable_categories),
        },
        "counts": {
            "history_events": len(state.history),
            "future_confirmed_events": len(state.future_confirmed),
            "pending_debits": len(state.pending_debits),
            "pending_credits": len(state.pending_credits),
            "patterns": len(state.patterns),
            "income_streams": len(state.income_streams),
            "expense_streams": len(state.expense_streams),
            "conflicts_resolved": len(state.conflict_log),
        },
        "income_streams": stream_json(state.income_streams, is_expense=False),
        "expense_streams": stream_json(state.expense_streams, is_expense=True),
    }
