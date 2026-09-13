"""Test request_04 to see the pattern."""
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

request = data['sample_requests'][data['sample_requests']['request_id'] == 'request_04'].iloc[0]
print(f"Request: {request['request_id']}")
print(f"Date: {request['request_date'].date()}")
print(f"Requested: {request['requested_amount']:,.0f}")
print(f"Expected safe amount: 27,156.80")

state = financial_engine.reconstruct_state(request['user_id'], request['request_date'], request['request_id'])
print(f"\nBalance: {state.current_available_balance:,.2f}")
print(f"Minimum: {state.minimum_balance_to_keep:,.2f}")

forecast = financial_engine.forecast_balance(state, days=90)

# Show forecast around key dates
print(f"\nForecast (first few dates):")
dates_sorted = sorted(forecast.keys())[:10]
for date in dates_sorted:
    bal = forecast[date]
    marker = " <- REQUEST DATE" if date.date() == request['request_date'].date() else ""
    print(f"  {date.date()}: {bal:>15,.2f}{marker}")

# Test the expected amount
test_amount = 27156.80
print(f"\nTesting expected amount {test_amount:,.2f}: {'SAFE' if financial_engine._is_amount_safe(test_amount, state.as_of_date, state.minimum_balance_to_keep, forecast) else 'UNSAFE'}")

# Binary search to find actual maximum
low, high = 0.0, float(request['requested_amount'])
precision = 0.01

while high - low > precision:
    mid = (low + high) / 2.0
    if financial_engine._is_amount_safe(mid, state.as_of_date, state.minimum_balance_to_keep, forecast):
        low = mid
    else:
        high = mid

print(f"\nBinary search result: {low:,.2f}")
print(f"Expected:                    27,156.80")
print(f"Difference:                  {abs(low - 27156.80):,.2f}")
