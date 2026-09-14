"""Decision service — affordability analysis and audit persistence."""
from __future__ import annotations

from sqlalchemy.orm import Session

from ..core.affordability import maximum_safe_payment, earliest_safe_full_payment_date
from ..core.models import (
    AffordabilityRequest, AffordabilityStatus, Assumptions, Evidence, EvidenceType,
    FinancialState, Recommendation,
)
from ..core.planner import generate_candidate_plans
from ..core.ranking import rank_plans
from ..core.verifier import verify_plan
from ..core.version import canonicalize
from ..db.models import Decision, User, VerificationResult
from .audit_service import record_audit_event


def analyze_request(
    db: Session,
    user: User,
    state: FinancialState,
    request: AffordabilityRequest,
    assumptions: Assumptions,
) -> Recommendation:
    """Run the deterministic decision pipeline and audit the result.

    1. Generate candidates.
    2. Verify each through independent simulation.
    3. Rank strictly the verified survivors.
    4. Compile the fact sheet and log the permanent audit trace.
    """
    plans = generate_candidate_plans(request, state, assumptions)

    verified = []
    failed_attempts = 0
    for plan in plans:
        v_result = verify_plan(plan, request, state, assumptions)
        if v_result.passed:
            verified.append((plan, v_result))
        else:
            failed_attempts += 1

    if not verified:
        # A plan with method=NOT_RECOMMENDED always "passes" trivially
        # if the candidate generator was forced to emit it (the generator
        # emits it when all else fails). If even that didn't happen, we build
        # an explicit empty one here.
        raise RuntimeError("No candidate passed verify_plan, not even NOT_RECOMMENDED")

    # Sort the survivors deterministically
    survivors = [p for p, _ in verified]
    ranked = rank_plans(survivors, request)
    best_plan = ranked[0]
    best_verification = next(v for p, v in verified if p.plan_id == best_plan.plan_id)

    # Determine top-level status
    status = AffordabilityStatus.NOT_AFFORDABLE
    if best_plan.method.value == "full_payment":
        status = AffordabilityStatus.AFFORDABLE_NOW
        if best_plan.spending_changes:
            status = AffordabilityStatus.AFFORDABLE_WITH_PLAN
    elif best_plan.method.value == "wait" or best_plan.method.value == "installments":
        status = AffordabilityStatus.AFFORDABLE_LATER
    elif best_plan.method.value == "partial_payment":
        status = AffordabilityStatus.AFFORDABLE_WITH_PLAN

    # Compile the fact sheet (the constrained evidence for the Phase 3 LLM)
    facts = list(best_verification.evidence)
    facts.append(Evidence(
        EvidenceType.REQUEST,
        source="user_request",
        fact=f"User requested {request.requested_amount.amount_minor} {request.requested_amount.currency}",
        amount=request.requested_amount,
        date=request.desired_completion_date,
    ))
    max_safe = maximum_safe_payment(state, assumptions, request.request_date)
    facts.append(Evidence(
        EvidenceType.SAFE_AMOUNT,
        source="engine",
        fact=f"Maximum safe payment on request date is {max_safe.amount_minor}",
        amount=max_safe,
        date=request.request_date,
    ))
    if max_safe < request.requested_amount:
        earliest = earliest_safe_full_payment_date(state, assumptions, request.requested_amount)
        if earliest:
            facts.append(Evidence(
                EvidenceType.EARLIEST_DATE,
                source="engine",
                fact=f"Amount becomes safe in full on {earliest.isoformat()}",
                date=earliest,
            ))

    rec = Recommendation(
        request_id=request.request_id,
        status=status,
        plan=best_plan,
        verification=best_verification,
        ranked_alternatives=tuple(p for p in ranked if p.plan_id != best_plan.plan_id),
        evidence=tuple(facts),
        engine_version=best_verification.engine_version,
        input_hash=best_verification.input_hash,
    )

    # -----------------------------------------------------------------------
    # Audit persistence
    # -----------------------------------------------------------------------
    vr_record = VerificationResult(
        user_id=user.id,
        request_id=request.request_id,
        plan_id=best_verification.plan_id,
        passed=True,
        violations=[],
        evidence=canonicalize(best_verification.evidence),
        engine_version=best_verification.engine_version,
        input_hash=best_verification.input_hash,
    )
    db.add(vr_record)

    db_decision = Decision(
        user_id=user.id,
        request_id=request.request_id,
        state_input_hash=rec.input_hash,
        engine_version=rec.engine_version,
        status=rec.status.value,
        selected_plan=canonicalize(best_plan),
        fact_sheet=canonicalize(facts),
        decision_layer=f"deterministic-engine-{rec.engine_version}",
    )
    db.add(db_decision)

    record_audit_event(
        db, event_type="decision_made",
        user_id=user.id, request_id=request.request_id,
        payload={
            "status": rec.status.value,
            "method": best_plan.method.value,
            "failed_attempts": failed_attempts,
            "rank_1_plan_id": best_plan.plan_id,
        }
    )
    db.flush()
    return rec
