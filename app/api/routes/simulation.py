from datetime import date
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .users import get_db

router = APIRouter()

@router.post("")
def run_simulation(
    user_id: str,
    horizon_days: int = 90,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Run a deterministic forecast over the current state and persist the series."""
    from ...services.user_service import get_user
    from ...services.forecast_service import run_forecast, _series_to_json

    user = get_user(db, user_id)
    today = date.today()
    series = run_forecast(db, user, as_of=today, horizon_days=horizon_days, persist=True)
    db.commit()

    return _series_to_json(series)
