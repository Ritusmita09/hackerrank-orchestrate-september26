"""Test the _is_amount_safe function directly."""
import sys
from datetime import datetime
sys.path.insert(0, 'code')

from data_loader import DataLoader
from image_extractor import ImageExtractor
from financial_engine import FinancialEngine

loader = DataLoader()
data = loader.load_all()
lookups = loader.build_lookup_structures(data)

image_extractor = ImageExtractor()
financial_engine = FinancialEngine(data, lookups, image_extractor)

request = data['sample_requests'][data['sample_requests']['request_id'] == 'request_03'].iloc[0]
state = financial_engine.reconstruct_state(request['user_id'], request['request_date'], request['request_id'])
forecast = financial_engine.forecast_balance(state, days=90)

# Test the expected safe amount
amount = 873000
print(f"Testing amount_safe_to_pay = {amount:,.0f}")
print(f"Request date: {state.as_of_date.date()}")
print(f"Minimum balance: {state.minimum_balance_to_keep:,.0f}")

# Manually trace through _is_amount_safe
print(f"\nChecking each date in forecast:")
all_safe = True
for date, balance in forecast.items():
    adjusted_balance = balance
    if date == state.as_of_date:
        adjusted_balance -= amount
        print(f"  {date.date()}: {balance:>12,.0f} - {amount:>12,.0f} = {adjusted_balance:>12,.0f}", end="")
    else:
        print(f"  {date.date()}: {balance:>12,.0f} (no payment)", end="")
    
    if adjusted_balance < state.minimum_balance_to_keep:
        print(f"  -> UNSAFE (below minimum {state.minimum_balance_to_keep:,.0f})")
        all_safe = False
        break
    else:
        print(f"  -> safe")

if all_safe:
    print(f"\nResult: SAFE")
else:
    print(f"\nResult: UNSAFE")

# Also test what the function returns
result = financial_engine._is_amount_safe(amount, state.as_of_date, state.minimum_balance_to_keep, forecast)
print(f"Function result: {'SAFE' if result else 'UNSAFE'}")
