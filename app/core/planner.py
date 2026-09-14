"""The Strategy Planner.

Generates candidate plans for an AffordabilityRequest based on the user's
FinancialState and explicitly-authorized options.

The planner generates candidates. It does NOT enforce final safety — that
is the job of the verification gate, which intercepts every recommended plan.
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta
from itertools import combinations
from typing import List, Optional, Tuple, Sequence

from .affordability import earliest_safe_full_payment_date, is_safe_payment, maximum_safe_payment
from .forecast import forecast
from .models import (
    AffordabilityRequest,
    Assumptions,
    EngineError,
    FinancialState,
    Money,
    Payment,
    PaymentMethod,
    PaymentOption,
    PaymentPlan,
    SimulationScenario,
    SpendingChange,
    SpendingChangeAction,
)


def _generate_plan_id(method: PaymentMethod, req_id: str, nonce: str) -> str:
    msg = f"{method.value}:{req_id}:{nonce}".encode("utf-8")
    return "plan-" + hashlib.sha1(msg).hexdigest()[:12]


def _build_full_payment(
    req: AffordabilityRequest, state: FinancialState, assumptions: Assumptions
) -> Optional[PaymentPlan]:
    if PaymentMethod.FULL_PAYMENT not in state.payment_methods_considered:
        return None

    # Ideally paying today. Can we afford it?
    date_ = state.as_of
    amount = req.requested_amount
    if is_safe_payment(state, assumptions, date_, amount):
        return PaymentPlan(
            plan_id=_generate_plan_id(PaymentMethod.FULL_PAYMENT, req.request_id, "opt"),
            method=PaymentMethod.FULL_PAYMENT,
            payments=(Payment(date_, amount),),
            total_payable=amount
        )
    return None


def _build_wait_payment(
    req: AffordabilityRequest, state: FinancialState, assumptions: Assumptions
) -> Optional[PaymentPlan]:
    if PaymentMethod.WAIT not in state.payment_methods_considered:
        return None

    amount = req.requested_amount
    earliest = earliest_safe_full_payment_date(state, assumptions, amount)

    # WAIT implies later than today, and must still meet the deadline
    if earliest and earliest > state.as_of and earliest <= req.desired_completion_date:
        return PaymentPlan(
            plan_id=_generate_plan_id(PaymentMethod.WAIT, req.request_id, earliest.isoformat()),
            method=PaymentMethod.WAIT,
            payments=(Payment(earliest, amount),),
            total_payable=amount
        )
    return None


def _build_partial_payment(
    req: AffordabilityRequest, state: FinancialState, assumptions: Assumptions
) -> Optional[PaymentPlan]:
    if PaymentMethod.PARTIAL_PAYMENT not in state.payment_methods_considered:
        return None
    if not req.allows_partial_payment:
        return None

    date_ = state.as_of
    max_safe_today = maximum_safe_payment(state, assumptions, date_)

    if max_safe_today.is_zero or max_safe_today.is_negative:
        return None
    if max_safe_today >= req.requested_amount:
        return None  # We can fully pay today, partial makes no sense

    remainder = req.requested_amount - max_safe_today

    # Simulate first payment having been made
    scenario = SimulationScenario(payments=(Payment(date_, max_safe_today),))
    try:
        series = forecast(state, assumptions, scenario)
    except EngineError:
        return None

    # We need to find the earliest date the remainder can be paid given the today payment.
    # We do a simpler scan for now
    current = date_ + timedelta(days=1)
    horizon_limit = state.as_of + timedelta(days=assumptions.horizon_days)

    while current <= req.desired_completion_date and current <= horizon_limit:
        test_scenario = SimulationScenario(
            payments=(Payment(date_, max_safe_today), Payment(current, remainder))
        )
        try:
            test_series = forecast(state, assumptions, test_scenario)
            min_bal = min(pt.closing_balance for pt in test_series.points)
            if min_bal >= state.minimum_balance:
                return PaymentPlan(
                    plan_id=_generate_plan_id(PaymentMethod.PARTIAL_PAYMENT, req.request_id, current.isoformat()),
                    method=PaymentMethod.PARTIAL_PAYMENT,
                    payments=(Payment(date_, max_safe_today), Payment(current, remainder)),
                    total_payable=req.requested_amount
                )
        except EngineError:
            pass
        current += timedelta(days=1)

    return None


def _build_installments(
    req: AffordabilityRequest, state: FinancialState, assumptions: Assumptions
) -> List[PaymentPlan]:
    if PaymentMethod.INSTALLMENTS not in state.payment_methods_considered:
        return []

    plans: List[PaymentPlan] = []

    for opt in req.payment_options:
        # A single-payment "installment option" is structurally a full
        # payment; the verifier requires INSTALLMENTS to have >= 2 payments.
        if opt.number_of_payments < 2:
            continue
        # Check explicit profile limitation
        if state.profile.max_installment_payments is not None:
            if opt.number_of_payments > state.profile.max_installment_payments:
                continue

        payments = opt.build_payments()
        if not payments:
            continue

        last_date = payments[-1].date
        if last_date > req.desired_completion_date:
            continue

        if (last_date - state.as_of).days > assumptions.horizon_days:
            continue

        # Optional: fast-fail if clearly unaffordable, though verify_plan will catch it
        scenario = SimulationScenario(payments=payments)

        try:
            test_series = forecast(state, assumptions, scenario)
            min_bal = min(pt.closing_balance for pt in test_series.points)
            if min_bal >= state.minimum_balance:
                plans.append(PaymentPlan(
                    plan_id=_generate_plan_id(PaymentMethod.INSTALLMENTS, req.request_id, opt.option_id),
                    method=PaymentMethod.INSTALLMENTS,
                    payments=payments,
                    total_payable=opt.total_payable,
                    option_id=opt.option_id
                ))
        except EngineError:
            pass

    return plans


def _build_spending_change_plans(
    req: AffordabilityRequest, state: FinancialState, assumptions: Assumptions
) -> List[PaymentPlan]:
    """Search for plans that become viable if the user cuts spending.

    We search subsets of authorized pattern cuts (size 1 to 3), rank them
    by total relief, and try to apply them to enable FULL/WAIT/PARTIAL.
    """
    # Build list of possible atomic changes
    possible_changes: List[SpendingChange] = []
    for stream in state.expense_streams:
        if stream.can_stop and stream.pattern:
            possible_changes.append(SpendingChange(
                action=SpendingChangeAction.STOP,
                pattern_id=stream.pattern.pattern_id,
                per_occurrence_relief=stream.typical_amount,
            ))
        elif stream.can_reduce and stream.pattern and stream.min_allowed_amount:
            # For reducible, we assume maximum possible reduction
            if stream.typical_amount > stream.min_allowed_amount:
                relief = stream.typical_amount - stream.min_allowed_amount
                possible_changes.append(SpendingChange(
                    action=SpendingChangeAction.REDUCE_TO,
                    pattern_id=stream.pattern.pattern_id,
                    per_occurrence_relief=relief,
                    new_amount=stream.min_allowed_amount
                ))

    if not possible_changes:
        return []

    plans: List[PaymentPlan] = []

    # Sort changes by single-occurrence relief (heuristic)
    possible_changes.sort(key=lambda c: c.per_occurrence_relief.amount_minor, reverse=True)

    amount = req.requested_amount
    date_ = state.as_of

    # Try combinations of size 1 and 2
    for size in (1, 2):
        for subset in combinations(possible_changes, size):
            changes = tuple(subset)

            # Can we afford full payment today with these changes?
            if PaymentMethod.FULL_PAYMENT in state.payment_methods_considered:
                scenario = SimulationScenario(
                    payments=(Payment(date_, amount),),
                    spending_changes=changes
                )
                try:
                    series = forecast(state, assumptions, scenario)
                    if min(pt.closing_balance for pt in series.points) >= state.minimum_balance:
                        # Success! We found a working spending change plan
                        change_ids = "+".join(c.pattern_id[:6] for c in changes)
                        plans.append(PaymentPlan(
                            plan_id=_generate_plan_id(PaymentMethod.FULL_PAYMENT, req.request_id, f"changes-{change_ids}"),
                            method=PaymentMethod.FULL_PAYMENT,
                            payments=(Payment(date_, amount),),
                            total_payable=amount,
                            spending_changes=changes
                        ))
                        # Don't keep generating more complex spending changes if a simple one works
                        continue
                except EngineError:
                    pass

    return plans


def generate_candidate_plans(
    req: AffordabilityRequest,
    state: FinancialState,
    assumptions: Assumptions
) -> List[PaymentPlan]:
    """Generate all valid candidate plans for the request."""
    candidates: List[PaymentPlan] = []

    # 1. Full Payment today
    full_plan = _build_full_payment(req, state, assumptions)
    if full_plan:
        candidates.append(full_plan)

    # 2. WAIT (pay in full, but later)
    wait_plan = _build_wait_payment(req, state, assumptions)
    if wait_plan:
        candidates.append(wait_plan)

    # 3. Partial (max today, rest later)
    partial_plan = _build_partial_payment(req, state, assumptions)
    if partial_plan:
        candidates.append(partial_plan)

    # 4. Installments
    candidates.extend(_build_installments(req, state, assumptions))

    # 5. Plans that require spending cuts
    # (Only attempt if we don't already have a clean FULL payment)
    if not full_plan:
        candidates.extend(_build_spending_change_plans(req, state, assumptions))

    # 6. Fallback: NOT_RECOMMENDED if no valid plan was generated
    if not candidates:
        candidates.append(PaymentPlan(
            plan_id=_generate_plan_id(PaymentMethod.NOT_RECOMMENDED, req.request_id, "fail"),
            method=PaymentMethod.NOT_RECOMMENDED,
            payments=(),
            total_payable=Money.zero(state.home_currency)
        ))

    return candidates
