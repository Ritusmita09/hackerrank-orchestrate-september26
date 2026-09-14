import unittest
from datetime import date, timedelta
from app.core.models import (
    Money, UserFinancialProfile, FinancialEvent, Direction, PaymentMethod,
    EventStatus, RecurringPattern, SimulationScenario, Payment,
    SpendingChange, SpendingChangeAction, EngineError
)
from app.core.state import BuildContext, build_state
from app.core.forecast import forecast
from app.core.models import Assumptions

class TestForecast(unittest.TestCase):
    def setUp(self):
        self.as_of = date(2026, 9, 14)
        self.USD = "USD"
        
        self.profile = UserFinancialProfile(
            user_id="u123",
            home_currency=self.USD,
            current_available_balance=Money.from_minor(500000, self.USD),
            minimum_balance_to_keep=Money.from_minor(100000, self.USD),
            payment_methods_considered=frozenset([PaymentMethod.FULL_PAYMENT]),
            protected_categories=frozenset(["housing", "groceries"]),
            stoppable_categories=frozenset(["entertainment"]),
            reducible_categories=frozenset(["dining_out"]),
            max_installment_payments=None
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
        
    def test_conservative_pending_balance(self):
        from app.core.models import EventStatus
        # Pending debits reduce the initial balance immediately, pending credits do not
        ctx = BuildContext(
            user_id="u123",
            as_of=self.as_of,
            profile=self.profile,
            events=[
                FinancialEvent("e1", self.as_of - timedelta(days=1), Direction.DEBIT, Money.from_minor(20000, self.USD), "category", "pending bill", EventStatus.PENDING),
                FinancialEvent("e2", self.as_of - timedelta(days=1), Direction.CREDIT, Money.from_minor(50000, self.USD), "category", "pending refund", EventStatus.PENDING),
            ],
            patterns=[]
        )
        state = build_state(ctx)
        
        series = forecast(state, self.assumptions)
        
        # Day 0 should start at 5000 and close at 4800 (only debit applied)
        pt0 = series.points[0]
        self.assertEqual(pt0.date, self.as_of)
        self.assertEqual(pt0.opening_balance.amount_minor, 500000)
        self.assertEqual(pt0.closing_balance.amount_minor, 480000)
        
        # If assumptions count pending credits, it closes at 5300
        assum2 = Assumptions(horizon_days=10, count_pending_credits=True)
        series2 = forecast(state, assum2)
        self.assertEqual(series2.points[0].closing_balance.amount_minor, 530000)
        
    def test_future_events(self):
        ctx = BuildContext(
            user_id="u123",
            as_of=self.as_of,
            profile=self.profile,
            events=[
                FinancialEvent("e1", self.as_of + timedelta(days=2), Direction.DEBIT, Money.from_minor(150000, self.USD), "housing", "rent", EventStatus.SCHEDULED),
                FinancialEvent("e2", self.as_of + timedelta(days=5), Direction.CREDIT, Money.from_minor(20000, self.USD), "shopping", "refund", EventStatus.SCHEDULED),
            ],
            patterns=[]
        )
        state = build_state(ctx)
        
        series = forecast(state, self.assumptions)
        
        self.assertEqual(series.points[2].closing_balance.amount_minor, 350000)
        self.assertEqual(series.points[4].closing_balance.amount_minor, 350000)
        self.assertEqual(series.points[5].closing_balance.amount_minor, 370000)
        
    def test_payment_scenario(self):
        scenario = SimulationScenario(
            payments=(Payment(self.as_of + timedelta(days=1), Money.from_minor(250000, self.USD)),)
        )
        
        series = forecast(self.state, self.assumptions, scenario)
        self.assertEqual(series.points[1].closing_balance.amount_minor, 250000)
        
    def test_payment_outside_horizon(self):
        scenario = SimulationScenario(
            payments=(Payment(self.as_of + timedelta(days=20), Money.from_minor(100, self.USD)),)
        )
        with self.assertRaises(EngineError):
            forecast(self.state, self.assumptions, scenario)

