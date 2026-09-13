"""Test the rent fix."""
import sys
from datetime import datetime, timedelta
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

print(f"Request date: {state.as_of_date.date()}")
print()

# Check what patterns were detected and when their next occurrence is
print("Recurring patterns and next occurrences:")
for pattern in state.recurring_patterns:
    next_date = pattern.next_occurrence(state.as_of_date)
    print(f"  {pattern.category}: next on {next_date.date()}, amount {pattern.typical_amount:,.0f}")

print()

# Look at forecast for first few days
print("Forecast for first 10 days:")
sorted_dates = sorted(forecast.keys())
for i in range(10):
    date = sorted_dates[i]
    balance = forecast[date]
    if i > 0:
        prev_balance = forecast[sorted_dates[i-1]]
        change = balance - prev_balance
        print(f"  {date.date()}: {balance:>12,.0f} (change: {change:>+10,.0f})")
    else:
        print(f"  {date.date()}: {balance:>12,.0f}")

# Find min balance in forecast
min_balance = min(forecast.values())
min_date = [d for d, b in forecast.items() if b == min_balance][0]
print(f"\nMinimum balance in forecast: {min_balance:,.0f} on {min_date.date()}")

# Calculate max safe amount
max_safe = min_balance - state.minimum_balance_to_keep
print(f"Minimum to keep: {state.minimum_balance_to_keep:,.0f}")
print(f"Max safe amount: {max_safe:,.0f}")
print(f"Expected: 873,000")
