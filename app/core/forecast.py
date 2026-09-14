"""Cash flow forecasting.

The forecast is a pure function:
    (FinancialState, Assumptions, SimulationScenario) -> CashflowSeries

This ensures the deterministic verifier can re-simulate any candidate plan
without touching global state, mocking components, or relying on the planner's
own arithmetic.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Set, Tuple

from .models import (
    Assumptions,
    CashFlow,
    CashflowSeries,
    Direction,
    FinancialState,
    ForecastPoint,
    SimulationScenario,
    Money,
    SpendingChangeAction,
)
from .recurrence import pattern_occurrences
from .version import ENGINE_VERSION, input_hash


def forecast(
    state: FinancialState,
    assumptions: Assumptions,
    scenario: SimulationScenario = SimulationScenario(),
) -> CashflowSeries:
    """Generate a daily balance forecast over the horizon.

    Base components (from state):
    - Starting balance = current balance - pending debits.
    - Pending credits are excluded (unless assumptions.count_pending_credits).
    - Future confirmed events (SCHEDULED status).
    - Future pattern occurrences, deduplicated against scheduled events on
      (stream_key, date, direction) so that seeing the salary hit "pending"
      two days early doesn't double-count against the pattern.

    Scenario components:
    - Payments: injected directly as cash flows.
    - Spending changes: REDUCE_TO replaces future pattern occurrences;
      STOP omits them entirely. A change to a stream deletes any SCHEDULED
      events on that stream inside the horizon (so cancelling a subscription
      voids the upcoming bill).
    """
    days = assumptions.horizon_days
    start_date = state.as_of
    end_date = start_date + timedelta(days=days)
    currency = state.home_currency

    # 1. Start balance and immediate pending reservations
    current_balance = state.current_balance
    start_flows: List[CashFlow] = []

    for debit in state.pending_debits:
        current_balance = current_balance - debit.amount
        start_flows.append(CashFlow(
            ref=f"pending:{debit.event_id}",
            direction=Direction.DEBIT,
            amount=debit.amount,
            description=f"Pending: {debit.description}"
        ))

    if assumptions.count_pending_credits:
        for credit in state.pending_credits:
            current_balance = current_balance + credit.amount
            start_flows.append(CashFlow(
                ref=f"pending_credit:{credit.event_id}",
                direction=Direction.CREDIT,
                amount=credit.amount,
                description=f"Pending credit: {credit.description}"
            ))

    # 2. Gather active scenario spending changes
    stopped_patterns: Set[str] = set()
    reduced_patterns: Dict[str, Money] = {}
    for change in scenario.spending_changes:
        if change.action is SpendingChangeAction.STOP:
            stopped_patterns.add(change.pattern_id)
        elif change.action is SpendingChangeAction.REDUCE_TO:
            if change.new_amount is None:
                raise ValueError(f"REDUCE_TO change on {change.pattern_id} lacks new_amount")
            reduced_patterns[change.pattern_id] = change.new_amount

    # 3. Future scheduled events
    flows_by_date: Dict[date, List[CashFlow]] = defaultdict(list)
    scheduled_keys: Set[Tuple[str, date, Direction]] = set()
    stopped_stream_keys: Set[str] = {
        p.stream_key for p in state.patterns if p.pattern_id in stopped_patterns
    }

    for event in state.future_confirmed:
        if not (start_date <= event.date <= end_date):
            continue
        # If the user stopped this stream entirely, the scheduled event is cancelled.
        if event.stream_key() in stopped_stream_keys:
            continue
        # (A REDUCE_TO change does NOT modify existing scheduled events; they
        # are assumed to represent a bill already cut. The reduction applies
        # only to the pattern-generated occurrences.)

        scheduled_keys.add((event.stream_key(), event.date, event.direction))
        flows_by_date[event.date].append(CashFlow(
            ref=f"event:{event.event_id}",
            direction=event.direction,
            amount=event.amount,
            description=event.description
        ))

    # 4. Generated pattern occurrences (deduplicated)
    for pattern in state.patterns:
        # Never forecast unconfirmed income.
        if pattern.direction is Direction.CREDIT and not pattern.user_confirmed:
            continue
        if pattern.pattern_id in stopped_patterns:
            continue

        occurrences = pattern_occurrences(pattern, start_date, end_date)
        for occ in occurrences:
            # Deduplicate against scheduled events.
            if (pattern.stream_key, occ.date, occ.direction) in scheduled_keys:
                continue

            amount = occ.amount
            if pattern.pattern_id in reduced_patterns:
                amount = reduced_patterns[pattern.pattern_id]

            flows_by_date[occ.date].append(CashFlow(
                ref=f"pattern:{pattern.pattern_id}:{occ.date.isoformat()}",
                direction=occ.direction,
                amount=amount,
                description=f"Generated: {occ.description}"
            ))

    # 5. Simulated payments
    for i, payment in enumerate(scenario.payments):
        if not (start_date <= payment.date <= end_date):
            from .models import EngineError
            raise EngineError(f"scenario payment {payment.date} is outside forecast horizon")
        flows_by_date[payment.date].append(CashFlow(
            ref=f"payment:{i}",
            direction=Direction.DEBIT,
            amount=payment.amount,
            description=f"Simulated payment {i+1}"
        ))

    # 6. Walk the timeline to produce balance points
    if start_flows:
        # Start-day flows are folded into the opening balance, so the
        # day-0 point.opening_balance explicitly shows how we got there.
        # However, for consistency we record them on start_date.
        flows_by_date[start_date] = start_flows + flows_by_date.get(start_date, [])

    points: List[ForecastPoint] = []
    current_date = start_date

    # The running balance starts at the pre-pending current balance.
    # On day 0, start_flows will apply.
    running = state.current_balance

    while current_date <= end_date:
        opening = running
        day_flows = flows_by_date.get(current_date, [])
        for flow in day_flows:
            if flow.direction is Direction.CREDIT:
                running = running + flow.amount
            else:
                running = running - flow.amount

        # Sort flows for deterministic output (credits first, then size, then ref)
        day_flows.sort(key=lambda f: (
            0 if f.direction is Direction.CREDIT else 1,
            -f.amount.amount_minor,
            f.ref
        ))

        points.append(ForecastPoint(
            date=current_date,
            opening_balance=opening,
            closing_balance=running,
            flows=tuple(day_flows)
        ))
        current_date = current_date + timedelta(days=1)

    return CashflowSeries(
        start_date=start_date,
        horizon_days=days,
        points=tuple(points),
        engine_version=ENGINE_VERSION,
        input_hash=input_hash(state, assumptions, scenario)
    )
