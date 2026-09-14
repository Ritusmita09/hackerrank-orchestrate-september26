"""The Verification Gate.

This is the system's safety invariant: no plan may be recommended to a user
unless `verify_plan(...)` explicitly returns `passed=True`.

The verifier does NOT trust the planner. It re-simulates the plan from scratch
against the canonical state and asserts every safety rule individually.
"""
from __future__ import annotations

from typing import List

from .forecast import forecast
from .models import (
    AffordabilityRequest,
    Assumptions,
    EngineError,
    Evidence,
    EvidenceType,
    FinancialState,
    PaymentMethod,
    PaymentPlan,
    SpendingChangeAction,
    VerificationResult,
    Violation,
    ViolationCode,
)
from .version import ENGINE_VERSION, input_hash


def verify_plan(
    plan: PaymentPlan,
    request: AffordabilityRequest,
    state: FinancialState,
    assumptions: Assumptions,
) -> VerificationResult:
    """Rigorous, independent verification of a candidate plan.

    Five conditions must be met:
    1. Legal bounds: payments are positive, dates are in bounds, totals sum up.
    2. Request matched: the payment total is exact, deadline is met.
    3. Method matched: the pattern of payments matches the declared strategy
       (e.g., partial means exactly two, wait means tomorrow+).
    4. Safety floor: the simulated daily balance NEVER drops below the user's
       ``minimum_balance_to_keep``.
    5. Authorized changes: spending changes target only patterns the user
       has explicitly marked stoppable/reducible.
    """
    violations: List[Violation] = []
    evidence: List[Evidence] = []

    # -----------------------------------------------------------------------
    # 1. Structural and Bounds Checks
    # -----------------------------------------------------------------------
    if plan.method is PaymentMethod.NOT_RECOMMENDED:
        # A not-recommended plan is not a runnable strategy; it passes
        # verification vacuously as an empty plan, but its semantics mean
        # "don't do this."
        if plan.payments:
            violations.append(Violation(
                ViolationCode.PAYMENT_STRUCTURE_INVALID,
                "NOT_RECOMMENDED plan cannot contain payments"
            ))
    elif not plan.payments:
        violations.append(Violation(
            ViolationCode.PAYMENT_STRUCTURE_INVALID,
            "Active plan contains no payments"
        ))

    computed_total = 0
    for i, p in enumerate(plan.payments):
        if not p.amount.is_positive:
            violations.append(Violation(
                ViolationCode.NEGATIVE_OR_ZERO_PAYMENT,
                f"Payment {i+1} amount {p.amount} is not positive",
                date=p.date, amount=p.amount
            ))
        if p.amount.currency != state.home_currency:
            violations.append(Violation(
                ViolationCode.CURRENCY_MISMATCH,
                f"Payment {i+1} currency {p.amount.currency} != {state.home_currency}",
                date=p.date
            ))
        if p.date < state.as_of:
            violations.append(Violation(
                ViolationCode.DATE_BEFORE_AS_OF,
                f"Payment {i+1} is in the past: {p.date} < {state.as_of}",
                date=p.date
            ))
        if (p.date - state.as_of).days > assumptions.horizon_days:
            violations.append(Violation(
                ViolationCode.DATE_BEYOND_HORIZON,
                f"Payment {i+1} {p.date} exceeds horizon {assumptions.horizon_days} days",
                date=p.date
            ))
        computed_total += p.amount.amount_minor

    if plan.payments and computed_total != plan.total_payable.amount_minor:
        violations.append(Violation(
            ViolationCode.PAYMENT_STRUCTURE_INVALID,
            f"Payments sum to {computed_total} but total_payable is {plan.total_payable.amount_minor}"
        ))

    # -----------------------------------------------------------------------
    # 2. Request Constraints
    # -----------------------------------------------------------------------
    if plan.method is not PaymentMethod.NOT_RECOMMENDED:
        if plan.last_payment_date and plan.last_payment_date > request.desired_completion_date:
            violations.append(Violation(
                ViolationCode.DEADLINE_MISSED,
                f"Plan completes {plan.last_payment_date} > deadline {request.desired_completion_date}",
                date=plan.last_payment_date
            ))

        # Check total satisfied (for non-installments, it must exactly match)
        if plan.method is not PaymentMethod.INSTALLMENTS:
            if computed_total != request.requested_amount.amount_minor:
                violations.append(Violation(
                    ViolationCode.OBLIGATION_NOT_SATISFIED,
                    f"Plan pays {computed_total} but request is {request.requested_amount.amount_minor}"
                ))

    # -----------------------------------------------------------------------
    # 3. Method Semantics
    # -----------------------------------------------------------------------
    if plan.method not in state.payment_methods_considered \
            and plan.method is not PaymentMethod.NOT_RECOMMENDED:
        violations.append(Violation(
            ViolationCode.METHOD_NOT_PERMITTED,
            f"Method {plan.method.value} not authorized by user profile"
        ))

    if plan.method is PaymentMethod.FULL_PAYMENT:
        if len(plan.payments) != 1:
            violations.append(Violation(
                ViolationCode.PAYMENT_STRUCTURE_INVALID,
                "FULL_PAYMENT must have exactly one payment"
            ))
    elif plan.method is PaymentMethod.PARTIAL_PAYMENT:
        if not request.allows_partial_payment:
            violations.append(Violation(
                ViolationCode.PARTIAL_NOT_PERMITTED,
                "Request does not permit partial payments"
            ))
        if len(plan.payments) != 2:
            violations.append(Violation(
                ViolationCode.PAYMENT_STRUCTURE_INVALID,
                "PARTIAL_PAYMENT must have exactly two payments"
            ))
    elif plan.method is PaymentMethod.WAIT:
        if len(plan.payments) != 1:
            violations.append(Violation(
                ViolationCode.PAYMENT_STRUCTURE_INVALID,
                "WAIT must have exactly one payment"
            ))
        if plan.first_payment_date == state.as_of:
            violations.append(Violation(
                ViolationCode.PAYMENT_STRUCTURE_INVALID,
                "WAIT payment cannot be today"
            ))
    elif plan.method is PaymentMethod.INSTALLMENTS:
        if len(plan.payments) < 2:
            violations.append(Violation(
                ViolationCode.PAYMENT_STRUCTURE_INVALID,
                "INSTALLMENTS must have >= 2 payments"
            ))
        if state.profile.max_installment_payments is not None:
            if len(plan.payments) > state.profile.max_installment_payments:
                violations.append(Violation(
                    ViolationCode.INSTALLMENT_LIMIT_EXCEEDED,
                    f"Option has {len(plan.payments)} payments > user limit "
                    f"{state.profile.max_installment_payments}"
                ))

        # Check that it structurally matches the registered PaymentOption
        if plan.option_id:
            opt = next(
                (o for o in request.payment_options if o.option_id == plan.option_id),
                None
            )
            if opt is None:
                violations.append(Violation(
                    ViolationCode.INSTALLMENT_MISMATCH,
                    f"Option {plan.option_id} not found in request"
                ))
            else:
                opt_payments = opt.build_payments()
                if len(plan.payments) != len(opt_payments):
                    violations.append(Violation(
                        ViolationCode.INSTALLMENT_MISMATCH,
                        f"Plan has {len(plan.payments)} payments, "
                        f"option has {len(opt_payments)}"
                    ))
                for i, (pp, op) in enumerate(zip(plan.payments, opt_payments)):
                    if pp.date != op.date or pp.amount != op.amount:
                        violations.append(Violation(
                            ViolationCode.INSTALLMENT_MISMATCH,
                            f"Payment {i+1} {pp.amount} on {pp.date} != "
                            f"option {op.amount} on {op.date}"
                        ))
        else:
            violations.append(Violation(
                ViolationCode.INSTALLMENT_MISMATCH,
                "INSTALLMENTS plan must specify option_id"
            ))

    # -----------------------------------------------------------------------
    # 4. Authorized Spending Changes
    # -----------------------------------------------------------------------
    pattern_index = {p.pattern_id: p for p in state.patterns}

    for change in plan.spending_changes:
        pattern = pattern_index.get(change.pattern_id)
        if not pattern:
            violations.append(Violation(
                ViolationCode.SPENDING_CHANGE_INVALID,
                f"Unknown pattern {change.pattern_id}"
            ))
            continue
        if pattern.category in state.protected_categories:
            violations.append(Violation(
                ViolationCode.SPENDING_CHANGE_UNAUTHORIZED,
                f"Pattern {pattern.pattern_id} belongs to protected category "
                f"{pattern.category}"
            ))
        if change.action is SpendingChangeAction.STOP:
            if pattern.category not in state.stoppable_categories:
                violations.append(Violation(
                    ViolationCode.SPENDING_CHANGE_UNAUTHORIZED,
                    f"User has not authorized stopping category {pattern.category}"
                ))
            evidence.append(Evidence(
                EvidenceType.SPENDING_CHANGE,
                source=f"pattern:{change.pattern_id}",
                fact=f"Stop {pattern.description or pattern.category} entirely"
            ))
        elif change.action is SpendingChangeAction.REDUCE_TO:
            if pattern.category not in state.reducible_categories:
                violations.append(Violation(
                    ViolationCode.SPENDING_CHANGE_UNAUTHORIZED,
                    f"User has not authorized reducing category {pattern.category}"
                ))
            if change.new_amount is None:
                violations.append(Violation(
                    ViolationCode.SPENDING_CHANGE_INVALID,
                    "REDUCE_TO lacks new_amount"
                ))
            elif pattern.min_allowed_amount is not None:
                if change.new_amount < pattern.min_allowed_amount:
                    violations.append(Violation(
                        ViolationCode.SPENDING_CHANGE_INVALID,
                        f"Reduced amount {change.new_amount} < minimum allowed "
                        f"{pattern.min_allowed_amount}"
                    ))
            evidence.append(Evidence(
                EvidenceType.SPENDING_CHANGE,
                source=f"pattern:{change.pattern_id}",
                fact=f"Reduce {pattern.description or pattern.category} to {change.new_amount}"
            ))

    # -----------------------------------------------------------------------
    # 5. Financial Safety (Dynamic Simulation)
    # -----------------------------------------------------------------------
    # If the plan is structurally invalid, forecasting it might crash or
    # mask errors. Only run the deep simulation if surface checks passed.
    if not violations and plan.method is not PaymentMethod.NOT_RECOMMENDED:
        try:
            series = forecast(state, assumptions, plan.as_scenario())
            min_bal = state.minimum_balance
            for pt in series.points:
                if pt.closing_balance < min_bal:
                    violations.append(Violation(
                        ViolationCode.MIN_BALANCE_BREACH,
                        f"Balance drops to {pt.closing_balance} on {pt.date}, "
                        f"below minimum {min_bal}",
                        date=pt.date, amount=pt.closing_balance
                    ))
                    break  # One breach is enough

            # Record the lowest point as explanation evidence
            lowest = series.min_point()
            evidence.append(Evidence(
                EvidenceType.MIN_BALANCE_FLOOR,
                source="verification_flight",
                fact=f"Lowest simulated balance is {lowest.closing_balance} on {lowest.date}",
                date=lowest.date, amount=lowest.closing_balance
            ))
        except EngineError as e:
            violations.append(Violation(
                ViolationCode.PAYMENT_STRUCTURE_INVALID,
                f"Simulation failed: {e}"
            ))

    # Compile the result
    return VerificationResult(
        plan_id=plan.plan_id,
        passed=len(violations) == 0,
        violations=tuple(violations),
        evidence=tuple(evidence),
        engine_version=ENGINE_VERSION,
        input_hash=input_hash(plan, request, state, assumptions)
    )
