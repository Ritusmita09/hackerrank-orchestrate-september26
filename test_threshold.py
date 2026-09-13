"""Test the threshold where request_03 becomes unsafe."""
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

# Test amounts around the expected threshold
test_amounts = [
    872000, 872500, 872900, 872950, 872990, 872999,
    873000, 873001, 873010, 873100, 873500, 874000
]

print("Testing amounts around expected threshold of 873,000:")
for amount in test_amounts:
    result = financial_engine._is_amount_safe(amount, state.as_of_date, state.minimum_balance_to_keep, forecast)
    status = "SAFE" if result else "UNSAFE"
    marker = " <-- EXPECTED MAX" if amount == 873000 else ""
    print(f"  {amount:>9,.0f}: {status}{marker}")

# Find the exact threshold using binary search with higher precision
print("\nFinding exact threshold with high precision:")
low, high = 0.0, float(request['requested_amount'])
precision = 0.0001  # Much higher precision

while high - low > precision:
    mid = (low + high) / 2.0
    if financial_engine._is_amount_safe(mid, state.as_of_date, state.minimum_balance_to_keep, forecast):
        low = mid
    else:
        high = mid

print(f"Threshold amount: {low:,.2f}")
print(f"Expected:         873,000.00")
print(f"Difference:       {abs(low - 873000):,.2f}")
