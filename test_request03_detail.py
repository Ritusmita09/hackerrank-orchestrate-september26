"""Detailed analysis of request_03."""
import sys
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
print(f"Request: {request['request_id']}")
print(f"Date: {request['request_date'].date()}")
print(f"Requested: {request['requested_amount']:,.0f}")
print(f"Expected safe amount: 873,000")
print(f"Expected earliest date: 2019-11-15")

state = financial_engine.reconstruct_state(request['user_id'], request['request_date'], request['request_id'])
print(f"\nBalance: {state.current_available_balance:,.0f}")
print(f"Minimum: {state.minimum_balance_to_keep:,.0f}")

forecast = financial_engine.forecast_balance(state, days=90)

# Show forecast around key dates
print(f"\nForecast around request date and expected date:")
import pandas as pd
from datetime import datetime
key_dates = [
    datetime(2019, 9, 3),   # Request date
    datetime(2019, 9, 15),  # +12 days
    datetime(2019, 9, 29),  # +26 days (current earliest)
    datetime(2019, 10, 15), # +42 days
    datetime(2019, 11, 1),  # +59 days
    datetime(2019, 11, 15), # +73 days (expected earliest)
]

for date in key_dates:
    if date in forecast:
        bal = forecast[date]
        marker = ""
        if date.date() == request['request_date'].date():
            marker = " <- REQUEST DATE"
        elif date.date() == datetime(2019, 11, 15).date():
            marker = " <- EXPECTED EARLIEST"
        print(f"  {date.date()}: {bal:>15,.0f}{marker}")

# Check what makes Nov 15 safer than Sep 29
print(f"\nIs 5,491,000 safe on 2019-09-29? ", end="")
safe_sep29 = financial_engine._is_amount_safe(request['requested_amount'], datetime(2019, 9, 29), state.minimum_balance_to_keep, forecast)
print(safe_sep29)

print(f"Is 5,491,000 safe on 2019-11-15? ", end="")
safe_nov15 = financial_engine._is_amount_safe(request['requested_amount'], datetime(2019, 11, 15), state.minimum_balance_to_keep, forecast)
print(safe_nov15)

# Check minimum balance in each forecast window
from datetime import timedelta
for payment_date_str, expected in [("2019-09-29", False), ("2019-11-15", True)]:
    payment_date = datetime.strptime(payment_date_str, "%Y-%m-%d")
    print(f"\nIf paying on {payment_date.date()}:")
    min_bal_after = min([bal - request['requested_amount'] if date == payment_date else bal 
                         for date, bal in forecast.items()])
    print(f"  Minimum balance after payment: {min_bal_after:,.0f}")
    print(f"  Required minimum: {state.minimum_balance_to_keep:,.0f}")
    print(f"  Passes: {min_bal_after >= state.minimum_balance_to_keep}")
