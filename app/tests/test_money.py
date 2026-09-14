import unittest
from decimal import Decimal
from app.core.models import Money, InvalidMoneyError, CurrencyMismatchError

class TestMoney(unittest.TestCase):
    def test_from_major(self):
        m = Money.from_major("12.34", "USD")
        self.assertEqual(m.amount_minor, 1234)
        self.assertEqual(m.currency, "USD")
        
        m2 = Money.from_major(100, "JPY")
        self.assertEqual(m2.amount_minor, 100)
        
    def test_reject_floats(self):
        with self.assertRaises(InvalidMoneyError):
            Money.from_major(12.34, "USD")
            
    def test_precision_bounds(self):
        with self.assertRaises(InvalidMoneyError):
            Money.from_major("12.345", "USD")
            
    def test_arithmetic(self):
        m1 = Money.from_minor(1000, "USD")
        m2 = Money.from_minor(500, "USD")
        
        self.assertEqual((m1 + m2).amount_minor, 1500)
        self.assertEqual((m1 - m2).amount_minor, 500)
        
    def test_currency_mismatch(self):
        m1 = Money.from_minor(1000, "USD")
        m2 = Money.from_minor(1000, "EUR")
        
        with self.assertRaises(CurrencyMismatchError):
            _ = m1 + m2
            
        with self.assertRaises(CurrencyMismatchError):
            _ = m1 > m2
