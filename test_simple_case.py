"""Test _is_amount_safe with a simple, known case."""
import sys
from datetime import datetime, timedelta
sys.path.insert(0, 'code')

from data_loader import DataLoader
from image_extractor import ImageExtractor
from financial_engine import FinancialEngine

# Create a mock financial engine with simple data
class MockFinancialEngine(FinancialEngine):
    def __init__(self):
        # Don't call parent __init__ to avoid loading data
        pass
    
    def _is_amount_safe(self, amount: float, payment_date: datetime,
                       minimum_balance: float, forecast: dict) -> bool:
        """Copy of the original function for testing."""
        for date, balance in forecast.items():
            adjusted_balance = balance
            if date == payment_date:
                adjusted_balance -= amount

            if adjusted_balance < minimum_balance:
                return False
        return True

# Test case 1: Simple case where forecast shows constant balance
print("Test 1: Constant balance forecast")
forecast1 = {
    datetime(2020, 1, 1): 10000.0,  # payment date
    datetime(2020, 1, 2): 10000.0,
    datetime(2020, 1, 3): 10000.0,
}
min_balance = 5000.0
payment_date = datetime(2020, 1, 1)

engine = MockFinancialEngine()
# Should be safe to pay up to 5000 (10000 - 5000)
max_safe = 10000.0 - 5000.0
print(f"Maximum safe amount should be: {max_safe}")
print(f"Testing {max_safe}: {'SAFE' if engine._is_amount_safe(max_safe, payment_date, min_balance, forecast1) else 'UNSAFE'}")
print(f"Testing {max_safe + 0.01}: {'SAFE' if engine._is_amount_safe(max_safe + 0.01, payment_date, min_balance, forecast1) else 'UNSAFE'}")
print()

# Test case 2: Forecast with future lower balance
print("Test 2: Forecast with future lower balance")
forecast2 = {
    datetime(2020, 1, 1): 15000.0,  # payment date
    datetime(2020, 1, 2): 8000.0,   # future date with lower balance
    datetime(2020, 1, 3): 12000.0,
}
min_balance = 5000.0
payment_date = datetime(2020, 1, 1)

# Even though payment date balance is high, future date drops to 8000
# But since we only adjust the payment date balance, the future date of 8000 is still >= 5000
# So we should still be able to pay up to 10000 on the payment date
max_safe = 15000.0 - 5000.0
print(f"Maximum safe amount should be: {max_safe}")
print(f"Testing {max_safe}: {'SAFE' if engine._is_amount_safe(max_safe, payment_date, min_balance, forecast2) else 'UNSAFE'}")
print(f"Testing {max_safe + 0.01}: {'SAFE' if engine._is_amount_safe(max_safe + 0.01, payment_date, min_balance, forecast2) else 'UNSAFE'}")
print()

# Test case 3: Forecast where payment date is the minimum
print("Test 3: Payment date has minimum balance")
forecast3 = {
    datetime(2020, 1, 1): 8000.0,   # payment date - lowest balance
    datetime(2020, 1, 2): 10000.0,
    datetime(2020, 1, 3): 12000.0,
}
min_balance = 5000.0
payment_date = datetime(2020, 1, 1)

# Payment date balance is 8000, minimum is 5000, so we can pay up to 3000
max_safe = 8000.0 - 5000.0
print(f"Maximum safe amount should be: {max_safe}")
print(f"Testing {max_safe}: {'SAFE' if engine._is_amount_safe(max_safe, payment_date, min_balance, forecast3) else 'UNSAFE'}")
print(f"Testing {max_safe + 0.01}: {'SAFE' if engine._is_amount_safe(max_safe + 0.01, payment_date, min_balance, forecast3) else 'UNSAFE'}")
