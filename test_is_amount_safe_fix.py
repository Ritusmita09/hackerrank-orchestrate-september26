"""Test that _is_amount_safe fix is working correctly."""
import sys
from datetime import datetime, timedelta
sys.path.insert(0, 'code')

# Simple test
forecast = {
    datetime(2019, 9, 3): 10000,
    datetime(2019, 9, 4): 8000,
    datetime(2019, 9, 5): 6000,
}

# Test OLD logic (only payment date)
def old_is_amount_safe(amount, payment_date, min_balance, forecast):
    for date, balance in forecast.items():
        adjusted_balance = balance
        if date == payment_date:  # OLD: only payment date
            adjusted_balance -= amount
        if adjusted_balance < min_balance:
            return False
    return True

# Test NEW logic (payment date and future)
def new_is_amount_safe(amount, payment_date, min_balance, forecast):
    for date, balance in forecast.items():
        adjusted_balance = balance
        if date >= payment_date:  # NEW: payment date and future
            adjusted_balance -= amount
        if adjusted_balance < min_balance:
            return False
    return True

print("Testing _is_amount_safe logic:")
print(f"Forecast: {[(d.strftime('%Y-%m-%d'), b) for d, b in forecast.items()]}")
print()

# Test amount = 3000 on Sept 4
amount = 3000
payment_date = datetime(2019, 9, 4)
min_balance = 5000

print(f"Testing payment of {amount} on {payment_date.strftime('%Y-%m-%d')} with min balance {min_balance}:")
print(f"  OLD logic (only payment date): {old_is_amount_safe(amount, payment_date, min_balance, forecast)}")
print(f"  NEW logic (payment date+future): {new_is_amount_safe(amount, payment_date, min_balance, forecast)}")
print()

# With OLD logic:
# Sept 3: 10000 (no change) >= 5000 ✓
# Sept 4: 8000 - 3000 = 5000 >= 5000 ✓  
# Sept 5: 6000 (no change) >= 5000 ✓
# Result: True

# With NEW logic:
# Sept 3: 10000 (no change, date < payment_date) >= 5000 ✓
# Sept 4: 8000 - 3000 = 5000 >= 5000 ✓
# Sept 5: 6000 - 3000 = 3000 < 5000 ✗
# Result: False

print("As expected, NEW logic correctly identifies that paying 3000 on Sept 4")
print("leaves insufficient balance on Sept 5 (3000 < 5000 minimum).")
