"""Affordability checks driven by exact integer binary search.

Every candidate amount is verified independently against the strict
rules of the deterministic engine by running a full forecast.

We use monotonically precise algorithms (exact integer arithmetic) to
eliminate floating point rounding defects and guarantee that if
``is_safe_payment`` is True, the verifier will pass it.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Tuple

from .forecast import forecast
from .models import (
    Assumptions,
    EngineError,
    FinancialState,
    Money,
    Payment,
    SimulationScenario,
)


def is_safe_payment(
    state: FinancialState,
    assumptions: Assumptions,
    date_: date,
    amount: Money,
) -> bool:
    """Return True if paying ``amount`` on ``date_`` respects the minimum balance.

    This runs a full simulated forecast from ``state.as_of`` to horizon end,
    injecting the candidate payment. If the simulated daily balance drops
    below ``state.minimum_balance`` at any point, the payment is unsafe.
    """
    if date_ < state.as_of:
        return False
    if (date_ - state.as_of).days > assumptions.horizon_days:
        return False
    if amount.is_negative:
        return False
    if amount.currency != state.home_currency:
        return False

    scenario = SimulationScenario(payments=(Payment(date_, amount),))
    try:
        series = forecast(state, assumptions, scenario)
    except EngineError:
        return False

    min_bal = state.minimum_balance
    for pt in series.points:
        if pt.closing_balance < min_bal:
            return False
    return True


def maximum_safe_payment(
    state: FinancialState,
    assumptions: Assumptions,
    date_: date,
) -> Money:
    """Find the exact largest amount that can be paid on ``date_`` safely.

    Uses an exact binary search over integer minor units. The search bounds
    are [0, state.current_balance].
    """
    if date_ < state.as_of:
        return Money.zero(state.home_currency)
    if (date_ - state.as_of).days > assumptions.horizon_days:
        return Money.zero(state.home_currency)

    # Fast baseline check: if a zero payment fails, the state is already breached.
    if not is_safe_payment(state, assumptions, date_, Money.zero(state.home_currency)):
        return Money.zero(state.home_currency)

    # Optional optimization: we could find a tighter upper bound (e.g. by
    # running a $0 forecast and finding the minimum headroom), but binary
    # search over `current_balance` takes ~50 steps for $100M down to pennies.
    # We choose the simpler, robust upper bound.
    # An edge case is if a massive incoming scheduled credit makes the max safe
    # payment LARGER than current_balance. To be safely monotonic, we use the
    # maximum closing balance of the 0-payment forecast.
    try:
        baseline_series = forecast(state, assumptions, SimulationScenario())
    except EngineError:
        return Money.zero(state.home_currency)

    max_forecasted_balance = max(pt.closing_balance for pt in baseline_series.points)

    low = 0
    high = max_forecasted_balance.amount_minor
    best = 0

    while low <= high:
        mid = low + (high - low) // 2
        candidate = Money.from_minor(mid, state.home_currency)
        if is_safe_payment(state, assumptions, date_, candidate):
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return Money.from_minor(best, state.home_currency)


def earliest_safe_full_payment_date(
    state: FinancialState,
    assumptions: Assumptions,
    amount: Money,
) -> Optional[date]:
    """Find the earliest date where ``amount`` can be paid in full safely.

    Checks every calendar day inside the forecast horizon.
    Returns None if the amount cannot be paid safely on any date.
    """
    if not amount.is_positive:
        return state.as_of
    if amount.currency != state.home_currency:
        return None

    days = assumptions.horizon_days
    current = state.as_of

    for _ in range(days + 1):
        if is_safe_payment(state, assumptions, current, amount):
            return current
        current = current + timedelta(days=1)

    return None
