import unittest
from datetime import date, timedelta
from app.core.models import (
    Money, UserFinancialProfile, FinancialEvent, Direction, PaymentMethod,
    EventStatus, RecurringPattern, Assumptions, PaymentPlan, Payment,
    AffordabilityRequest, PaymentOption, SpendingChange, SpendingChangeAction
)
from app.core.state import BuildContext, build_state
from app.core.verifier import verify_plan
from app.core.models import ViolationCode

class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.as_of = date(2026, 9, 14)
        self.USD = "USD"
        
        self.profile = UserFinancialProfile(
            user_id="u123",
            home_currency=self.USD,
            current_available_balance=Money.from_minor(300000, self.USD),
            minimum_balance_to_keep=Money.from_minor(100000, self.USD),
            payment_methods_considered=frozenset([
                PaymentMethod.FULL_PAYMENT, 
                PaymentMethod.INSTALLMENTS,
                PaymentMethod.PARTIAL_PAYMENT
            ]),
            max_installment_payments=4
        )
        ctx = BuildContext(
            user_id="u123",
            as_of=self.as_of,
            profile=self.profile,
            events=[],
            patterns=[]
        )
        self.state = build_state(ctx)
        self.assumptions = Assumptions(horizon_days=10)
        
        # We owe $500 today, deadline in 5 days
        self.request = AffordabilityRequest(
            request_id="r1", request_date=self.as_of,
            requested_amount=Money.from_minor(50000, self.USD),
            desired_completion_date=self.as_of + timedelta(days=5),
            allows_partial_payment=True,
            payment_options=()
        )
        
    def test_valid_full_payment(self):
        plan = PaymentPlan(
            plan_id="p1",
            method=PaymentMethod.FULL_PAYMENT,
            payments=(Payment(self.as_of, Money.from_minor(50000, self.USD)),),
            total_payable=Money.from_minor(50000, self.USD)
        )
        res = verify_plan(plan, self.request, self.state, self.assumptions)
        self.assertTrue(res.passed, f"Violations: {res.violations}")
        
    def test_wrong_method(self):
        # We don't have WAIT authorized
        plan = PaymentPlan(
            plan_id="p1",
            method=PaymentMethod.WAIT,
            payments=(Payment(self.as_of + timedelta(days=2), Money.from_minor(50000, self.USD)),),
            total_payable=Money.from_minor(50000, self.USD)
        )
        res = verify_plan(plan, self.request, self.state, self.assumptions)
        self.assertFalse(res.passed)
        self.assertTrue(any(v.code == ViolationCode.METHOD_NOT_PERMITTED for v in res.violations))
        
    def test_unsatisfied_obligation(self):
        plan = PaymentPlan(
            plan_id="p1",
            method=PaymentMethod.FULL_PAYMENT,
            payments=(Payment(self.as_of, Money.from_minor(40000, self.USD)),),
            total_payable=Money.from_minor(40000, self.USD)
        )
        res = verify_plan(plan, self.request, self.state, self.assumptions)
        self.assertFalse(res.passed)
        self.assertTrue(any(v.code == ViolationCode.OBLIGATION_NOT_SATISFIED for v in res.violations))
        
    def test_past_deadline(self):
        plan = PaymentPlan(
            plan_id="p2",
            method=PaymentMethod.INSTALLMENTS, # bypassing single-day rule to test deadline
            payments=(
                Payment(self.as_of, Money.from_minor(25000, self.USD)),
                Payment(self.as_of + timedelta(days=6), Money.from_minor(25000, self.USD))
            ),
            total_payable=Money.from_minor(50000, self.USD),
            option_id="opt1"
        )
        # Even if Option matching was OK, it misses the deadline
        res = verify_plan(plan, self.request, self.state, self.assumptions)
        self.assertFalse(res.passed)
        self.assertTrue(any(v.code == ViolationCode.DEADLINE_MISSED for v in res.violations))
        
    def test_min_balance_breach(self):
        # Headroom is $2000. Try to pay $2500.
        req_2500 = AffordabilityRequest("r2", Money.from_minor(250000, self.USD), self.as_of, self.as_of, True, ())
        plan = PaymentPlan(
            plan_id="p3",
            method=PaymentMethod.FULL_PAYMENT,
            payments=(Payment(self.as_of, Money.from_minor(250000, self.USD)),),
            total_payable=Money.from_minor(250000, self.USD)
        )
        res = verify_plan(plan, req_2500, self.state, self.assumptions)
        self.assertFalse(res.passed)
        self.assertTrue(any(v.code == ViolationCode.MIN_BALANCE_BREACH for v in res.violations))

