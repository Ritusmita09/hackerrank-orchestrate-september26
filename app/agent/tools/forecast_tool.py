import datetime
from datetime import date
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.db.models import User
from app.core.models import SimulationScenario, FinancialEvent, Money, EventStatus, Direction
from app.services.forecast_service import run_forecast, _series_to_json
from app.agent.tools.base import BaseTool

class ScenarioPaymentArgs(BaseModel):
    date: datetime.date = Field(..., description="The date of the hypothetical payment.")
    amount_minor: int = Field(..., description="The payment amount in minor units (e.g. cents).")

class ForecastToolArgs(BaseModel):
    horizon_days: int = Field(365, description="Number of days to forecast into the future.")
    scenario_payments: Optional[List[ScenarioPaymentArgs]] = Field(
        None, description="Hypothetical payments (debits) to include in the simulation."
    )

class ForecastTool(BaseTool):
    name = "run_forecast"
    description = "Run a deterministic financial simulation to forecast balances and cashflows over time."
    args_schema = ForecastToolArgs

    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        from app.core.models import Payment

        scenario_payments_args = args.get("scenario_payments") or []
        payments = []
        for p_args in scenario_payments_args:
            payments.append(Payment(
                date=p_args["date"],
                amount=Money(p_args["amount_minor"], user.home_currency),
            ))

        scenario = SimulationScenario(payments=tuple(payments))

        series = run_forecast(
            db,
            user,
            as_of=as_of,
            horizon_days=args["horizon_days"],
            scenario=scenario,
            persist=False  # Re-running simulations as part of LLM tool calling doesn't necessarily need to be persisted to avoid filling DB with hallucinated runs.
        )

        return _series_to_json(series)
