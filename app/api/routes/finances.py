from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .users import get_db
from ..schemas import ProfileResponse

router = APIRouter()


@router.get("/financial-state", response_model=None)
def get_financial_state_view(user_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """The fully reconstructed deterministic state the engine would consume."""
    from ...services.user_service import get_user
    from ...services.financial_state_service import get_financial_state, state_to_json

    user = get_user(db, user_id)
    state = get_financial_state(db, user, as_of=date.today())
    return state_to_json(state)


@router.get("/financial-state/profile", response_model=ProfileResponse)
def get_user_profile(user_id: str, db: Session = Depends(get_db)):
    """The latest active financial profile for the user."""
    from ...services.user_service import get_user, get_latest_profile
    user = get_user(db, user_id)
    profile = get_latest_profile(db, user.id)
    return ProfileResponse(
        revision=profile.revision,
        current_balance_minor=profile.current_balance_minor,
        minimum_balance_minor=profile.minimum_balance_minor,
        currency=profile.currency,
        protected_categories=profile.protected_categories or [],
        payment_methods_considered=profile.payment_methods_considered or [],
    )


# ---------------------------------------------------------------------------
# Recurring patterns — proposals await a user decision (Phase 4 confirmation
# flows); only user-confirmed, active patterns influence forecasts.
# ---------------------------------------------------------------------------

class PatternResponse(BaseModel):
    id: str
    stream_key: str
    category: str
    direction: str
    period_type: str
    period_days: Optional[int]
    period_days_of_month: list
    typical_amount_minor: int
    estimator: str
    currency: str
    last_seen: date
    evidence_event_ids: list
    confidence: float
    user_confirmed: bool
    active: bool
    description: str
    flexibility: str
    min_allowed_amount_minor: Optional[int]
    created_at: Optional[str]


@router.get("/patterns", response_model=list[PatternResponse])
def list_patterns_route(
    user_id: str,
    confirmed: Optional[bool] = None,
    include_inactive: bool = False,
    db: Session = Depends(get_db),
):
    """The user's recurring patterns.

    ``confirmed=false`` surfaces the unconfirmed proposals awaiting a user
    decision; by default both proposals and confirmed patterns are listed
    (inactive/rejected ones only with ``include_inactive=true``).
    """
    from ...services.user_service import get_user
    from ...services.pattern_service import list_patterns, pattern_to_json

    user = get_user(db, user_id)
    return [
        pattern_to_json(row)
        for row in list_patterns(
            db, user, confirmed=confirmed, include_inactive=include_inactive
        )
    ]


@router.post("/patterns/{pattern_id}/confirm", response_model=PatternResponse)
def confirm_pattern_route(
    user_id: str,
    pattern_id: str,
    db: Session = Depends(get_db),
):
    """Make a detected recurring-stream proposal authoritative."""
    from ...services.user_service import get_user
    from ...services.pattern_service import pattern_to_json
    from ...services.confirmation_service import (
        ConfirmationServiceError,
        confirm_pattern,
    )

    user = get_user(db, user_id)
    try:
        row = confirm_pattern(db, user, pattern_id=pattern_id)
        db.commit()
    except ConfirmationServiceError as exc:
        db.rollback()
        status = 404 if "not_found" in exc.code else 400
        raise HTTPException(status_code=status, detail=exc.message) from None

    return pattern_to_json(row)


@router.post("/patterns/{pattern_id}/reject", response_model=PatternResponse)
def reject_pattern_route(
    user_id: str,
    pattern_id: str,
    db: Session = Depends(get_db),
):
    """Retire a pattern (proposal or confirmed): it stops driving forecasts.

    Past transactions are untouched — rejection only stops future occurrences
    being generated from the pattern.
    """
    from ...services.user_service import get_user
    from ...services.pattern_service import pattern_to_json
    from ...services.confirmation_service import (
        ConfirmationServiceError,
        reject_pattern,
    )

    user = get_user(db, user_id)
    try:
        row = reject_pattern(db, user, pattern_id=pattern_id)
        db.commit()
    except ConfirmationServiceError as exc:
        db.rollback()
        status = 404 if "not_found" in exc.code else 400
        raise HTTPException(status_code=status, detail=exc.message) from None

    return pattern_to_json(row)
