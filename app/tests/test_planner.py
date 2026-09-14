import unittest
from datetime import date, timedelta
from app.core.models import (
    Money, UserFinancialProfile, FinancialEvent, Direction, PaymentMethod,
    EventStatus, RecurringPattern, Assumptions, PaymentPlan, Payment,
    AffordabilityRequest, PaymentOption, SpendingChange, SpendingChangeAction,
    ExpenseStream, MonthlyPeriod, Estimator
)
from app.core.state import BuildContext, build_state
from app.core.planner import generate_candidate_plans
from app.core.models import ViolationCode

class TestPlanner(unittest.TestCase):
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
                PaymentMethod.WAIT,
                PaymentMethod.PARTIAL_PAYMENT,
                PaymentMethod.INSTALLMENTS
            ]),
            max_installment_payments=None,
            stoppable_categories=frozenset(["entertainment"])
        )
        ctx = BuildContext(
            user_id="u123",
            as_of=self.as_of,
            profile=self.profile,
            events=[
                # Rent drops safe balance to $500
                FinancialEvent(
                    "e1", self.as_of + timedelta(days=2), 
                    Direction.DEBIT, Money.from_minor(150000, self.USD),
                    "housing", "rent", EventStatus.SCHEDULED
                ),
            ],
            patterns=[
                RecurringPattern(
                    "p1", "movies", "entertainment", Direction.DEBIT,
                    MonthlyPeriod((15,)), Money.from_minor(60000, self.USD),
                    Estimator.MEDIAN, self.as_of - timedelta(days=1), ("1",), 1.0, True
                )
            ]
        )
        self.state = build_state(ctx)
        self.assumptions = Assumptions(horizon_days=30)
        
    def test_generates_wait_and_partial(self):
        # Request $1200. Max safe today is $500. So FULL fails.
        # But wait... there's no upcoming income to makeWAIT viable.
        # We need an income event so WAIT hits.
        ctx = BuildContext(
            user_id="u123",
            as_of=self.as_of,
            profile=self.profile,
            events=[
                FinancialEvent(
                    "e1", self.as_of + timedelta(days=2), 
                    Direction.DEBIT, Money.from_minor(150000, self.USD),
                    "housing", "rent", EventStatus.SCHEDULED
                ),
                # Paycheck on day 4
                FinancialEvent(
                    "e2", self.as_of + timedelta(days=4), 
                    Direction.CREDIT, Money.from_minor(200000, self.USD),
                    "income", "paycheck", EventStatus.SCHEDULED
                ),
            ],
            patterns=[]
        )
        state_with_income = build_state(ctx)
        
        req = AffordabilityRequest(
            request_id="r1", 
            requested_amount=Money.from_minor(100000, self.USD),
            request_date=self.as_of,
            desired_completion_date=self.as_of + timedelta(days=10),
            allows_partial_payment=True,
            payment_options=()
        )
        
        plans = generate_candidate_plans(req, state_with_income, self.assumptions)
        methods = {p.method for p in plans}
        
        # Max safe today is $500. $1000 fails today.
        self.assertNotIn(PaymentMethod.FULL_PAYMENT, methods)
        
        # We can WAIT until day 4 (paycheck). Wait should exist.
        self.assertIn(PaymentMethod.WAIT, methods)
        
        # We can PARTIAL today ($500) and later ($500 on day 4). Partial should exist.
        self.assertIn(PaymentMethod.PARTIAL_PAYMENT, methods)

    def test_spending_cuts(self):
        # We are exactly at the edge. current 3000, rent 1500, min 1000. 
        # So we have exactly 500 headroom.
        # Let's say we have an entertainment pattern of $400 happening tomorrow.
        # So headroom drops to 100.
        # If we request $300, FULL_PAYMENT will fail unless we cut the entertainment.
        
        ctx = BuildContext(
            user_id="u123",
            as_of=self.as_of,
            profile=self.profile,
            events=[
                FinancialEvent("e1", self.as_of + timedelta(days=2), Direction.DEBIT, Money.from_minor(150000, self.USD), "housing", "rent", EventStatus.SCHEDULED),
            ],
            patterns=[
                RecurringPattern("p1", "fun", "entertainment", Direction.DEBIT, MonthlyPeriod((15,)), Money.from_minor(40000, self.USD), Estimator.MEDIAN, self.as_of - timedelta(days=1), ("1",), 1.0, True, min_allowed_amount=Money.from_minor(0, self.USD))
            ]
        )
        state = build_state(ctx)
        req = AffordabilityRequest("r2", Money.from_minor(30000, self.USD), self.as_of, self.as_of + timedelta(days=5), False, ())
        
        plans = generate_candidate_plans(req, state, self.assumptions)
        methods = {p.method for p in plans}
        
        # If spending changes logic works, it should find a FULL_PAYMENT that includes spending_changes.
        self.assertIn(PaymentMethod.FULL_PAYMENT, methods)
        
        full_plans = [p for p in plans if p.method == PaymentMethod.FULL_PAYMENT]
        self.assertTrue(len(full_plans[0].spending_changes) > 0)
        
