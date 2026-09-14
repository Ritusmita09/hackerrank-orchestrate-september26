from datetime import date
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .users import get_db
from ..schemas import AffordabilityRequestSchema, DecisionResponse

router = APIRouter()

@router.post("", response_model=DecisionResponse)
def make_decision(
    user_id: str,
    req: AffordabilityRequestSchema,
    db: Session = Depends(get_db)
):
    """Run the deterministic verification pipeline to answer an affordability question."""
    from ...services.user_service import get_user
    from ...services.financial_state_service import get_financial_state
    from ...services.decision_service import analyze_request
    from ...core.models import AffordabilityRequest, Assumptions, PaymentOption, Money
    from ...db.models import AffordabilityRequest as DBAffordabilityRequest

    user = get_user(db, user_id)
    today = date.today()  # True physical 'now' for the request date
    state = get_financial_state(db, user, as_of=today)

    # Reify the engine request
    options = []
    for o in req.payment_options:
        options.append(PaymentOption(
            option_id=o.option_id,
            first_payment_date=o.first_payment_date,
            frequency_days=o.frequency_days,
            number_of_payments=o.number_of_payments,
            payment_amount=Money.from_minor(o.payment_amount_minor, user.home_currency),
            total_payable=Money.from_minor(o.total_payable_minor, user.home_currency),
        ))

    db_req = DBAffordabilityRequest(
        user_id=user.id,
        requested_amount_minor=req.requested_amount_minor,
        currency=user.home_currency,
        request_date=today,
        desired_completion_date=req.desired_completion_date,
        allows_partial_payment=req.allows_partial_payment,
    )
    db.add(db_req)
    db.flush()

    engine_req = AffordabilityRequest(
        request_id=db_req.id,
        requested_amount=Money.from_minor(req.requested_amount_minor, user.home_currency),
        request_date=today,
        desired_completion_date=req.desired_completion_date,
        allows_partial_payment=req.allows_partial_payment,
        payment_options=tuple(options)
    )

    from ...core.version import canonicalize

    assumptions = Assumptions(horizon_days=90)
    rec = analyze_request(db, user, state, engine_req, assumptions)
    db.commit()

    return DecisionResponse(
        request_id=db_req.id,
        status=rec.status.value,
        plan_method=rec.plan.method.value if rec.plan else "not_recommended",
        total_payable_minor=rec.plan.total_payable.amount_minor if rec.plan else 0,
        engine_version=rec.engine_version,
        fact_sheet=canonicalize(rec.evidence),
    )
