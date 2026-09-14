import unittest
from datetime import date
from app.core.models import (
    Money, FinancialEvent, Direction, EventStatus, 
    FixedPeriod, MonthlyPeriod, RecurringPattern, Estimator, Flexibility
)
from app.core.recurrence import typical_amount_for, pattern_occurrences

class TestRecurrence(unittest.TestCase):
    def test_typical_amount_income_conservative(self):
        # 3 events, income -> uses MIN
        events = [
            FinancialEvent("1", date(2026, 1, 1), Direction.CREDIT, Money.from_minor(2000, "USD"), "inc"),
            FinancialEvent("2", date(2026, 2, 1), Direction.CREDIT, Money.from_minor(1800, "USD"), "inc"),
            FinancialEvent("3", date(2026, 3, 1), Direction.CREDIT, Money.from_minor(2200, "USD"), "inc"),
        ]
        amt, est = typical_amount_for(events, "USD")
        self.assertEqual(est, Estimator.MIN)
        self.assertEqual(amt.amount_minor, 1800)
        
    def test_typical_amount_expense_median(self):
        events = [
            FinancialEvent("1", date(2026, 1, 1), Direction.DEBIT, Money.from_minor(100, "USD"), "exp"),
            FinancialEvent("2", date(2026, 2, 1), Direction.DEBIT, Money.from_minor(130, "USD"), "exp"),
            FinancialEvent("3", date(2026, 3, 1), Direction.DEBIT, Money.from_minor(150, "USD"), "exp"),
            FinancialEvent("4", date(2026, 4, 1), Direction.DEBIT, Money.from_minor(170, "USD"), "exp"),
        ]
        amt, est = typical_amount_for(events, "USD")
        self.assertEqual(est, Estimator.MEDIAN)
        self.assertEqual(amt.amount_minor, 140)

    def test_monthly_clamping(self):
        pattern = RecurringPattern(
            pattern_id="p1",
            stream_key="rent",
            category="rent",
            direction=Direction.DEBIT,
            period=MonthlyPeriod(days_of_month=(31,)),
            typical_amount=Money.from_minor(1000, "USD"),
            estimator=Estimator.MEDIAN,
            last_seen=date(2025, 12, 31),
            evidence_event_ids=("1",),
            confidence=1.0,
            user_confirmed=True
        )
        occs = pattern_occurrences(pattern, date(2026, 1, 1), date(2026, 5, 1))
        dates = [o.date for o in occs]
        self.assertIn(date(2026, 1, 31), dates)
        self.assertIn(date(2026, 2, 28), dates)
        self.assertIn(date(2026, 3, 31), dates)
        self.assertIn(date(2026, 4, 30), dates)

    def test_weekly_frequency(self):
        pattern = RecurringPattern(
            pattern_id="p2",
            stream_key="coffee",
            category="coffee",
            direction=Direction.DEBIT,
            period=FixedPeriod(days=7),
            typical_amount=Money.from_minor(500, "USD"),
            estimator=Estimator.MEDIAN,
            last_seen=date(2025, 12, 29), # A Monday
            evidence_event_ids=("1",),
            confidence=1.0,
            user_confirmed=True
        )
        occs = pattern_occurrences(pattern, date(2026, 1, 1), date(2026, 1, 15))
        dates = [o.date for o in occs]
        self.assertIn(date(2026, 1, 5), dates)
        self.assertIn(date(2026, 1, 12), dates)
        self.assertEqual(len(dates), 2)
        
