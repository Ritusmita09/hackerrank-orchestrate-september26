"""Property-based tests for the deterministic engine.

Dependency-free property testing: a seeded PRNG (fixed seed, so runs are
reproducible and CI-stable) generates hundreds of randomized inputs, and
each test asserts an algebraic invariant that must hold for ANY input:

* Money — exact integer arithmetic, major-unit round-trips, total order.
* Recurrence — occurrence series are strictly increasing and in range.
* Forecast — the ledger identity (closing = opening + flows) holds every
  day, the series is complete over the horizon, and the function is pure
  (same inputs -> same points and same input hash).
* Affordability — safety is monotone in amount, and the binary search
  returns the exact boundary (max is safe, max + one minor unit is not).
* Planner soundness — every candidate the planner emits on coherent
  inputs passes the independent verification gate.
* Verifier soundness — a passing verification implies an independently
  re-simulated forecast never breaches the minimum-balance floor.
* Ranking — the sort is a strict total order: permutation-invariant and
  idempotent.

No external libraries (e.g. hypothesis): the engine layer stays at zero
non-stdlib dependencies, per the roadmap.
"""
import random
import unittest
from datetime import date, timedelta

from app.core.affordability import is_safe_payment, maximum_safe_payment
from app.core.forecast import forecast
from app.core.models import (
    AffordabilityRequest,
    Assumptions,
    Direction,
    Estimator,
    EventStatus,
    FinancialEvent,
    FixedPeriod,
    MonthlyPeriod,
    Money,
    Payment,
    PaymentMethod,
    PaymentOption,
    PaymentPlan,
    RecurringPattern,
    SimulationScenario,
    UserFinancialProfile,
)
from app.core.planner import generate_candidate_plans
from app.core.ranking import rank_plans
from app.core.recurrence import pattern_occurrences
from app.core.state import BuildContext, build_state
from app.core.verifier import verify_plan

SEED = 20260914
MONEY_TRIALS = 300
TRIALS = 60
CURRENCY = "USD"
AS_OF = date(2026, 9, 14)

CATEGORIES = ("housing", "groceries", "entertainment", "dining", "income", "shopping")


# ---------------------------------------------------------------------------
# Random input generators (coherent by construction)
# ---------------------------------------------------------------------------


def _random_profile(rng: random.Random) -> UserFinancialProfile:
    methods = {PaymentMethod.FULL_PAYMENT}
    for method in (PaymentMethod.WAIT, PaymentMethod.PARTIAL_PAYMENT,
                   PaymentMethod.INSTALLMENTS):
        if rng.random() < 0.7:
            methods.add(method)
    return UserFinancialProfile(
        user_id="prop-user",
        home_currency=CURRENCY,
        current_available_balance=Money.from_minor(rng.randrange(0, 2_000_000), CURRENCY),
        minimum_balance_to_keep=Money.from_minor(rng.randrange(0, 300_000), CURRENCY),
        payment_methods_considered=frozenset(methods),
        protected_categories=frozenset(["housing", "groceries"]),
        stoppable_categories=frozenset(["entertainment"]),
        reducible_categories=frozenset(["dining"]),
    )


def _random_events(rng: random.Random, count: int):
    events = []
    for i in range(count):
        offset = rng.randrange(-90, 60)
        if offset < 0:
            status = EventStatus.SETTLED
        elif rng.random() < 0.7:
            status = EventStatus.SCHEDULED
        elif rng.random() < 0.5:
            status = EventStatus.PENDING
        else:
            status = EventStatus.CANCELLED
        events.append(FinancialEvent(
            event_id=f"e{i}",
            date=AS_OF + timedelta(days=offset),
            direction=rng.choice((Direction.CREDIT, Direction.DEBIT)),
            amount=Money.from_minor(rng.randrange(1, 400_000), CURRENCY),
            category=rng.choice(CATEGORIES),
            description=f"stream {rng.randrange(0, 5)}",
            status=status,
        ))
    return events


def _random_pattern(rng: random.Random, index: int) -> RecurringPattern:
    if rng.random() < 0.5:
        period = FixedPeriod(rng.randrange(1, 31))
    else:
        anchors = {rng.randrange(1, 32)}
        if rng.random() < 0.3:
            anchors.add(rng.randrange(1, 32))
        period = MonthlyPeriod(tuple(sorted(anchors)))
    direction = rng.choice((Direction.CREDIT, Direction.DEBIT))
    category = rng.choice(CATEGORIES)
    min_allowed = None
    if direction is Direction.DEBIT and rng.random() < 0.5:
        min_allowed = Money.from_minor(rng.randrange(0, 20_000), CURRENCY)
    return RecurringPattern(
        pattern_id=f"p{index}",
        stream_key=f"prop stream {index} |{category}|{direction.value}",
        category=category,
        direction=direction,
        period=period,
        typical_amount=Money.from_minor(rng.randrange(1, 200_000), CURRENCY),
        estimator=rng.choice((Estimator.MIN, Estimator.P10, Estimator.MEDIAN)),
        last_seen=AS_OF - timedelta(days=rng.randrange(0, 40)),
        evidence_event_ids=(),
        confidence=round(rng.uniform(0.3, 1.0), 4),
        user_confirmed=rng.random() < 0.8,
        description=f"prop stream {index}",
        min_allowed_amount=min_allowed,
    )


def _random_state(rng: random.Random):
    profile = _random_profile(rng)
    patterns = [_random_pattern(rng, i) for i in range(rng.randrange(0, 4))]
    ctx = BuildContext(
        user_id="prop-user",
        as_of=AS_OF,
        profile=profile,
        events=_random_events(rng, rng.randrange(0, 10)),
        patterns=patterns,
    )
    return build_state(ctx)


def _random_request(rng: random.Random) -> AffordabilityRequest:
    options = ()
    if rng.random() < 0.5:
        n_pay = rng.randrange(2, 5)  # single-payment options are not installments
        amount = Money.from_minor(rng.randrange(1, 100_000), CURRENCY)
        options = (PaymentOption(
            option_id="opt0",
            first_payment_date=AS_OF + timedelta(days=rng.randrange(0, 20)),
            frequency_days=rng.randrange(7, 31),
            number_of_payments=n_pay,
            payment_amount=amount,
            total_payable=Money.from_minor(amount.amount_minor * n_pay, CURRENCY),
        ),)
    return AffordabilityRequest(
        request_id="prop-req",
        requested_amount=Money.from_minor(rng.randrange(1, 600_000), CURRENCY),
        request_date=AS_OF,
        desired_completion_date=AS_OF + timedelta(days=rng.randrange(0, 80)),
        allows_partial_payment=rng.random() < 0.7,
        payment_options=options,
    )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestMoneyProperties(unittest.TestCase):
    def test_arithmetic_and_roundtrips(self):
        rng = random.Random(SEED)
        for _ in range(MONEY_TRIALS):
            a = rng.randrange(-10**12, 10**12)
            b = rng.randrange(-10**12, 10**12)
            ma, mb = Money.from_minor(a, CURRENCY), Money.from_minor(b, CURRENCY)

            # Exact integer arithmetic
            self.assertEqual((ma + mb).amount_minor, a + b)
            self.assertEqual((ma - mb).amount_minor, a - b)
            # Commutativity and inverse
            self.assertEqual(ma + mb, mb + ma)
            self.assertEqual((ma - mb) + mb, ma)
            # Total order matches the minor-unit order
            self.assertEqual(ma < mb, a < b)
            self.assertEqual(ma == mb, a == b)

            # Major-unit round trips (2-decimal and 0-decimal currencies)
            self.assertEqual(Money.from_major(ma.to_major(), CURRENCY), ma)
            self.assertEqual(Money.from_major(str(ma.to_major()), CURRENCY), ma)
            mj = Money.from_minor(a, "JPY")
            self.assertEqual(Money.from_major(mj.to_major(), "JPY"), mj)


class TestRecurrenceProperties(unittest.TestCase):
    def test_occurrences_ordered_in_range_and_deterministic(self):
        rng = random.Random(SEED)
        for i in range(MONEY_TRIALS):
            pattern = _random_pattern(rng, i)
            start = AS_OF + timedelta(days=rng.randrange(-30, 30))
            end = start + timedelta(days=rng.randrange(0, 120))

            occurrences = pattern_occurrences(pattern, start, end)
            dates = [o.date for o in occurrences]

            # Strictly increasing, no duplicates
            self.assertEqual(dates, sorted(dates))
            self.assertEqual(len(dates), len(set(dates)))
            for d in dates:
                self.assertTrue(start <= d <= end)
                self.assertGreater(d, pattern.last_seen)
            # Every occurrence carries the pattern's typical amount
            for o in occurrences:
                self.assertEqual(o.amount, pattern.typical_amount)

            # Determinism: identical call, identical result
            self.assertEqual(pattern_occurrences(pattern, start, end), occurrences)


class TestForecastProperties(unittest.TestCase):
    def test_ledger_identity_completeness_and_purity(self):
        rng = random.Random(SEED)
        for _ in range(TRIALS):
            state = _random_state(rng)
            assumptions = Assumptions(horizon_days=rng.randrange(5, 60))
            payments = tuple(
                Payment(
                    AS_OF + timedelta(days=rng.randrange(0, assumptions.horizon_days + 1)),
                    Money.from_minor(rng.randrange(1, 300_000), CURRENCY),
                )
                for _ in range(rng.randrange(0, 3))
            )
            scenario = SimulationScenario(payments=payments)

            series = forecast(state, assumptions, scenario)

            # Completeness: exactly one point per day, starting at as_of
            self.assertEqual(len(series.points), assumptions.horizon_days + 1)
            for i, pt in enumerate(series.points):
                self.assertEqual(pt.date, AS_OF + timedelta(days=i))

            # Ledger identity: closing = opening + net flow, chained days
            self.assertEqual(series.points[0].opening_balance, state.current_balance)
            previous = None
            for pt in series.points:
                self.assertEqual(pt.closing_balance, pt.opening_balance + pt.net_flow)
                if previous is not None:
                    self.assertEqual(pt.opening_balance, previous.closing_balance)
                previous = pt

            # Purity: same inputs produce identical outputs and hash
            again = forecast(state, assumptions, scenario)
            self.assertEqual(again.input_hash, series.input_hash)
            self.assertEqual(again.points, series.points)


class TestAffordabilityProperties(unittest.TestCase):
    def test_safety_monotone_and_max_is_exact_boundary(self):
        rng = random.Random(SEED)
        for _ in range(TRIALS):
            state = _random_state(rng)
            assumptions = Assumptions(horizon_days=rng.randrange(5, 60))
            day = AS_OF + timedelta(days=rng.randrange(0, assumptions.horizon_days + 1))

            # Monotonicity: any amount below a safe amount is also safe
            amount_minor = rng.randrange(0, 800_000)
            if is_safe_payment(state, assumptions, day,
                               Money.from_minor(amount_minor, CURRENCY)):
                smaller = Money.from_minor(rng.randrange(0, amount_minor + 1), CURRENCY)
                self.assertTrue(
                    is_safe_payment(state, assumptions, day, smaller),
                    f"{smaller} should be safe when {amount_minor} minor is"
                )

            # Exact boundary: max is safe (when positive), max + 1 minor is not
            maximum = maximum_safe_payment(state, assumptions, day)
            one_more = Money.from_minor(maximum.amount_minor + 1, CURRENCY)
            self.assertFalse(
                is_safe_payment(state, assumptions, day, one_more),
                f"{one_more} must be unsafe when max safe is {maximum}"
            )
            if maximum.amount_minor > 0:
                self.assertTrue(
                    is_safe_payment(state, assumptions, day, maximum),
                    f"maximum_safe_payment returned {maximum} but it is not safe"
                )


class TestPlannerAndVerifierProperties(unittest.TestCase):
    def test_every_planner_candidate_passes_verification(self):
        """Planner soundness: on coherent inputs no candidate is emitted that
        the independent gate would reject, and every gate pass is backed by
        an independently re-simulated safe forecast."""
        rng = random.Random(SEED)
        verified = 0
        for _ in range(TRIALS):
            state = _random_state(rng)
            assumptions = Assumptions(horizon_days=rng.randrange(15, 90))
            request = _random_request(rng)

            plans = generate_candidate_plans(request, state, assumptions)
            # The planner always returns at least NOT_RECOMMENDED
            self.assertTrue(plans)

            for plan in plans:
                result = verify_plan(plan, request, state, assumptions)
                self.assertTrue(
                    result.passed,
                    f"planner emitted {plan.method} plan that failed the gate: "
                    f"{[v.code.value for v in result.violations]}"
                )
                if plan.method is PaymentMethod.NOT_RECOMMENDED:
                    continue

                # Independent recomputation of what the gate asserted
                series = forecast(state, assumptions, plan.as_scenario())
                for pt in series.points:
                    self.assertGreaterEqual(pt.closing_balance, state.minimum_balance)
                self.assertLessEqual(
                    plan.last_payment_date, request.desired_completion_date
                )
                verified += 1

        self.assertGreater(verified, 0, "property never exercised a real plan")

    def test_verifier_is_deterministic(self):
        rng = random.Random(SEED)
        for _ in range(TRIALS):
            state = _random_state(rng)
            assumptions = Assumptions(horizon_days=rng.randrange(15, 90))
            request = _random_request(rng)
            plans = generate_candidate_plans(request, state, assumptions)

            for plan in plans:
                first = verify_plan(plan, request, state, assumptions)
                second = verify_plan(plan, request, state, assumptions)
                self.assertEqual(first, second)


class TestRankingProperties(unittest.TestCase):
    def test_permutation_invariant_and_idempotent(self):
        rng = random.Random(SEED)
        for trial in range(TRIALS):
            plans = []
            for i in range(rng.randrange(1, 8)):
                n_pay = rng.randrange(0, 4)
                payments = tuple(
                    Payment(
                        AS_OF + timedelta(days=rng.randrange(0, 60)),
                        Money.from_minor(rng.randrange(1, 100_000), CURRENCY),
                    )
                    for _ in range(n_pay)
                )
                total = Money.from_minor(
                    sum(p.amount.amount_minor for p in payments), CURRENCY
                )
                plans.append(PaymentPlan(
                    plan_id=f"plan{trial}-{i}",
                    method=rng.choice(tuple(PaymentMethod)),
                    payments=payments,
                    total_payable=total,
                ))
            request = _random_request(rng)

            baseline = rank_plans(plans, request)
            self.assertEqual(len(baseline), len(plans))

            # Order is independent of input permutation
            for _ in range(3):
                shuffled = plans[:]
                rng.shuffle(shuffled)
                self.assertEqual(rank_plans(shuffled, request), baseline)

            # Sorting an already-sorted sequence changes nothing
            self.assertEqual(rank_plans(list(baseline), request), baseline)


if __name__ == "__main__":
    unittest.main()
