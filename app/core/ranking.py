"""Ranking algorithms for candidate payment plans.

The ranking order is deterministic and lexicographic:
1. Completes by desired completion date? (bool, True > False)
2. Requires no spending changes? (bool, True > False)
3. Total amount paid (Money, smaller is better)
4. First payment date (date, earlier is better)
5. Number of payments (int, fewer is better)
6. Option ID (string, for tie-breaking deterministic output)
"""
from __future__ import annotations

from typing import Sequence, Tuple

from .models import AffordabilityRequest, PaymentMethod, PaymentPlan


def _rank_key(plan: PaymentPlan, request: AffordabilityRequest) -> Tuple:
    """Produce a lexicographic tuple for sorting a PaymentPlan.

    Python sorts tuples left-to-right. We want the best plan to be at index 0,
    so we negate indicators where "larger" means "better" (e.g. booleans).
    """
    if plan.method is PaymentMethod.NOT_RECOMMENDED:
        # Puts NOT_RECOMMENDED at the absolute bottom; the plan-id tiebreak
        # keeps the order a strict total order (permutation-invariant).
        return (
            True,           # is_not_recommended (False > True because True sorts after False)
            0, 0, 0, 0, 0,  # paddings
            plan.plan_id,   # tiebreak (same shape as the normal key)
        )

    # 1. Completes by desired completion date?
    # We want True to sort BEFORE False. In Python, False (0) < True (1).
    # So we negate it: -True = -1, -False = 0. -1 < 0, so True sorts first.
    completes_on_time = False
    if plan.last_payment_date:
        completes_on_time = plan.last_payment_date <= request.desired_completion_date
    rank_completes = -int(completes_on_time)

    # 2. Requires no spending changes?
    no_spending_changes = len(plan.spending_changes) == 0
    rank_no_changes = -int(no_spending_changes)

    # 3. Total amount paid (smaller is better, naturally sorts correct)
    rank_total = plan.total_payable.amount_minor

    # 4. First payment date (earlier is better, naturally sorts correct)
    # Using ordinal so it sorts as a number. If no payment, push to bottom.
    if plan.first_payment_date:
        rank_start = plan.first_payment_date.toordinal()
    else:
        rank_start = 9999999

    # 5. Number of payments (fewer is better)
    rank_count = len(plan.payments)

    # 6. Option ID / Plan ID Tie breaker (alphabetical)
    rank_tiebreak = plan.option_id if plan.option_id else plan.plan_id

    return (
        False,           # not NOT_RECOMMENDED
        rank_completes,
        rank_no_changes,
        rank_total,
        rank_start,
        rank_count,
        rank_tiebreak
    )


def rank_plans(
    plans: Sequence[PaymentPlan],
    request: AffordabilityRequest
) -> Tuple[PaymentPlan, ...]:
    """Sort candidate plans best-first according to the domain rules."""
    return tuple(sorted(plans, key=lambda p: _rank_key(p, request)))
