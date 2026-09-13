"""Test when full payment becomes safe."""
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
print(f"Request: {request['request_id']}")
print(f"Requested amount: {request['requested_amount']:,.0f}")
print(f"Expected earliest date for full payment: 2019-11-15")

state = financial_engine.reconstruct_state(request['user_id'], request['request_date'], request['request_id'])
forecast = financial_engine.forecast_balance(state, days=90)

# Test paying full amount on various dates
test_dates = [
    datetime(2019, 9, 3),   # Request date
    datetime(2019, 9, 15),
    datetime(2019, 9, 29),
    datetime(2019, 10, 15),
    datetime(2019, 10, 29),
    datetime(2019, 11, 1),
    datetime(2019, 11, 15), # Expected earliest
    datetime(2019, 11, 30),
]

print(f"\nTesting safety of paying full amount ({request['requested_amount']:,.0f}) on various dates:")
for date in test_dates:
    if date in forecast:
        balance = forecast[date]
        # Check if paying full amount on this date is safe
        # We need to check if balance - amount >= minimum for ALL dates in forecast
        # (under the current implementation, only the payment date balance is reduced)
        safe = financial_engine._is_amount_safe(
            request['requested_amount'], 
            date, 
            state.minimum_balance_to_keep, 
            forecast
        )
        status = "SAFE" if safe else "UNSAFE"
        marker = " <- REQUEST DATE" if date == state.as_of_date else ""
        marker += " <- EXPECTED EARLIEST" if date.date() == datetime(2019, 11, 15).date() else ""
        print(f"  {date.date()}: balance={balance:>12,.0f} -> {status}{marker}")
    else:
        print(f"  {date.date()}: not in forecast")

# Also, let's find the actual earliest date using the financial engine's method
earliest = financial_engine.find_earliest_full_payment_date(
    state, request['requested_amount'], forecast
)
print(f"\nFinancial engine's earliest date for full payment: {earliest.date() if earliest else 'None'}")
print(f"Expected: 2019-11-15")
