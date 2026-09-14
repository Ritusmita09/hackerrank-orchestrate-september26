"""Forecast service — simulations over reconstructed state, persisted."""
from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from ..core.forecast import forecast
from ..core.models import Assumptions, CashflowSeries, SimulationScenario
from ..core.version import canonicalize
from ..db.models import ForecastRun, User
from .audit_service import record_audit_event
from .financial_state_service import get_financial_state


def _series_to_json(series: CashflowSeries) -> dict:
    """Compact JSON for the forecast_runs table (dates, balances, flows)."""
    return {
        "start_date": series.start_date.isoformat(),
        "horizon_days": series.horizon_days,
        "engine_version": series.engine_version,
        "input_hash": series.input_hash,
        "points": [
            {
                "date": p.date.isoformat(),
                "opening_minor": p.opening_balance.amount_minor,
                "closing_minor": p.closing_balance.amount_minor,
                "flows": [
                    {
                        "ref": f.ref,
                        "direction": f.direction.value,
                        "amount_minor": f.amount.amount_minor,
                        "description": f.description,
                    }
                    for f in p.flows
                ],
            }
            for p in series.points
        ],
    }


def run_forecast(
    db: Session,
    user: User,
    as_of: date,
    horizon_days: int,
    scenario: SimulationScenario | None = None,
    request_id: str | None = None,
    persist: bool = True,
) -> CashflowSeries:
    """Reconstruct state, run the pure forecast, and record the run."""
    state = get_financial_state(db, user, as_of)
    assumptions = Assumptions(horizon_days=horizon_days)
    series = forecast(state, assumptions, scenario or SimulationScenario())

    if persist:
        db.add(ForecastRun(
            user_id=user.id,
            request_id=request_id,
            assumptions=canonicalize(assumptions),
            engine_version=series.engine_version,
            input_hash=series.input_hash,
            series=_series_to_json(series),
        ))
        record_audit_event(
            db, event_type="forecast_run",
            user_id=user.id, request_id=request_id,
            payload={
                "horizon_days": horizon_days,
                "engine_version": series.engine_version,
                "input_hash": series.input_hash,
            },
        )
        db.flush()
    return series
