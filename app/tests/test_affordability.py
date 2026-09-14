import unittest
from datetime import date, timedelta
from app.core.models import (
    Money, UserFinancialProfile, FinancialEvent, Direction, PaymentMethod,
    EventStatus, RecurringPattern, Assumptions
)
from app.core.state import BuildContext, build_state
from app.core.affordability import (
    is_safe_payment, maximum_safe_payment, earliest_safe_full_payment_date
)

class TestAffordability(unittest.TestCase):
    def setUp(self):
        self.as_of = date(2026, 9, 14)
        self.USD = "USD"
        
        self.profile = UserFinancialProfile(
            user_id="u123",
            home_currency=self.USD,
            current_available_balance=Money.from_minor(300000, self.USD), # $3000
            minimum_balance_to_keep=Money.from_minor(100000, self.USD), # $1000
            payment_methods_considered=frozenset([PaymentMethod.FULL_PAYMENT]),
        )
        
        # Scenario: current balance $3000, min balance $1000.
        # So right now, maximum safe payment without any events is $2000.
        # But we'll add an upcoming event that changes things.
        ctx = BuildContext(
            user_id="u123",
            as_of=self.as_of,
            profile=self.profile,
            events=[
                # $1500 upcoming rent in 3 days.
                FinancialEvent(
                    "e1", self.as_of + timedelta(days=3), 
                    Direction.DEBIT, Money.from_minor(150000, self.USD),
                    "housing", "rent", EventStatus.SCHEDULED
                ),
                # $1000 upcoming credit in 5 days.
                FinancialEvent(
                    "e2", self.as_of + timedelta(days=5), 
                    Direction.CREDIT, Money.from_minor(100000, self.USD),
                    "income", "paycheck", EventStatus.SCHEDULED
                )
            ],
            patterns=[]
        )
        self.state = build_state(ctx)
        self.assumptions = Assumptions(horizon_days=10)
        
    def test_is_safe_payment(self):
        # Paying $500 today leaves $2500 -> day 3 rent drops it to $1000 -> safe.
        safe1 = is_safe_payment(
            self.state, self.assumptions, self.as_of, Money.from_minor(50000, self.USD)
        )
        self.assertTrue(safe1)
        
        # Paying $600 today leaves $2400 -> day 3 rent drops it to $900 -> unsafe!
        safe2 = is_safe_payment(
            self.state, self.assumptions, self.as_of, Money.from_minor(60000, self.USD)
        )
        self.assertFalse(safe2)
        
    def test_maximum_safe_payment(self):
        # Max safe today is $500, because the $1500 rent in 3 days will pull us right to the minimum balance.
        max_pay = maximum_safe_payment(self.state, self.assumptions, self.as_of)
        self.assertEqual(max_pay.amount_minor, 50000)
        
        # Max safe on day 6 is $1500!
        # Because we'll have paid the $1500 rent (bal $1500), but received the $1000 paycheck (bal $2500).
        # Which means we have $1500 headroom over the $1000 min balance.
        max_pay_day6 = maximum_safe_payment(self.state, self.assumptions, self.as_of + timedelta(days=6))
        self.assertEqual(max_pay_day6.amount_minor, 150000)
        
    def test_earliest_safe_full_payment(self):
        # $500 is safe today.
        earliest_500 = earliest_safe_full_payment_date(
            self.state, self.assumptions, Money.from_minor(50000, self.USD)
        )
        self.assertEqual(earliest_500, self.as_of)
        
        # $1200 is NOT safe today, nor tomorrow (headroom is $500).
        # We need the salary to hit. Salary hits on day 5, bringing balance to $3000 - $1500 + $1000 = $2500.
        # Headroom becomes $1500. So we can pay $1200 on day 5.
        earliest_1200 = earliest_safe_full_payment_date(
            self.state, self.assumptions, Money.from_minor(120000, self.USD)
        )
        self.assertEqual(earliest_1200, self.as_of + timedelta(days=5))
        
        # $2000 is never safe in this window, because max headroom we ever reach is $1500.
        earliest_2000 = earliest_safe_full_payment_date(
            self.state, self.assumptions, Money.from_minor(200000, self.USD)
        )
        self.assertIsNone(earliest_2000)
        
