import uuid
import datetime
from datetime import date
from typing import Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.db.models import User
from app.core.models import AffordabilityRequest, Assumptions, Money
from app.services.decision_service import analyze_request
from app.services.financial_state_service import get_financial_state
from app.agent.tools.base import BaseTool

class AffordabilityToolArgs(BaseModel):
    requested_amount_minor: int = Field(..., description="The amount to evaluate for affordability in minor units (e.g. cents).")
    desired_completion_date: datetime.date = Field(..., description="The date by which the purchase/payment needs to be made.")
    allows_partial_payment: bool = Field(False, description="Whether the user is willing to make partial payments over time.")
    horizon_days: int = Field(365, description="The simulation horizon in days.")

class AffordabilityTool(BaseTool):
    name = "check_affordability"
    description = "Check if a specific purchase or financial commitment is affordable based on the current financial state."
    args_schema = AffordabilityToolArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        state = get_financial_state(db, user, as_of=as_of)

        request = AffordabilityRequest(
            request_id=uuid.uuid4().hex,
            requested_amount=Money(args["requested_amount_minor"], state.home_currency),
            request_date=as_of,
            desired_completion_date=args["desired_completion_date"],
            allows_partial_payment=args["allows_partial_payment"],
            payment_options=(),
        )
        assumptions = Assumptions(horizon_days=args["horizon_days"])

        recommendation = analyze_request(db, user, state, request, assumptions)

        # Format a clear structured response for the LLM, reusing the same
        # serialization the Phase 2 decision API uses (canonicalize keeps the
        # evidence machine-checkable).
        from app.core.version import canonicalize

        return {
            "status": recommendation.status.value,
            "plan_method": recommendation.plan.method.value if recommendation.plan else "not_recommended",
            "total_payable_minor": recommendation.plan.total_payable.amount_minor if recommendation.plan else 0,
            "fact_sheet": canonicalize(recommendation.evidence),
        }
