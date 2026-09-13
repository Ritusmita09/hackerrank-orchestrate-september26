"""In-memory hypothesis tests for amount_safe_to_pay mismatches.

H_A: salary termination ("Final employer payroll") -> no future salary pattern (user_05)
H_B: salary date follows most recent historical payday, not modal (user_07: 15th -> 23rd)
H_C: gig/variable income modeling (user_10): no-pattern vs median variants
Each test: reconstruct state, mutate recurring_patterns in memory, recompute forecast + A.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "code"))
import pandas as pd
from data_loader import DataLoader
from image_extractor import ImageExtractor
from financial_engine import FinancialEngine

loader = DataLoader()
data = loader.load_all()
lookups = loader.build_lookup_structures(data)
fe = FinancialEngine(data, lookups, ImageExtractor())
sr = {r['request_id']: r for r in data['sample_requests'].to_dict('records')}

def compute(rid, mutate=None):
    s = sr[rid]
    state = fe.reconstruct_state(s['user_id'], pd.Timestamp(s['request_date']), rid)
    if mutate:
        mutate(state)
    forecast = fe.forecast_balance(state, days=90)
    a = fe.calculate_safe_amount(state, s['requested_amount'], forecast)
    return a, state, forecast

def show(rid, label, a, expected):
    mark = "MATCH" if abs(a - expected) < 0.01 else ("close" if abs(a-expected) < 0.02*max(expected,1) else "")
    print(f"  {label:<55} A={a:>14,.2f}  expected={expected:>14,.2f}  {mark}")

# ---- H_A: user_05 salary termination ----
print("H_A request_05 (expected 737.00): drop salary pattern entirely")
def drop_salary(state):
    state.recurring_patterns = [p for p in state.recurring_patterns if p.category != 'salary']
a, _, _ = compute('request_05', drop_salary)
show('request_05', 'no salary pattern', a, 737.0)
a0, _, _ = compute('request_05')
show('request_05', 'baseline (current)', a0, 737.0)

# ---- H_B: user_07 salary date 15 -> 23 ----
print("\nH_B request_07 (expected 87,170.56): salary day-of-month -> 23")
def set_day_23(state):
    for p in state.recurring_patterns:
        if p.category == 'salary':
            p.days_of_month = [23]
            p.day_of_month = 23
a, _, _ = compute('request_07', set_day_23)
show('request_07', 'salary on 23rd', a, 87170.56)
a0, _, _ = compute('request_07')
show('request_07', 'baseline (15th)', a0, 87170.56)

# ---- H_C: user_10 gig income ----
print("\nH_C request_10 (expected 12,700.00): gig income variants")
a0, st0, _ = compute('request_10')
show('request_10', 'baseline', a0, 12700.0)
for p in st0.recurring_patterns:
    if p.category == 'salary':
        print(f"    (salary pattern: typ={p.typical_amount:,.2f} dom={p.days_of_month})")
a, _, _ = compute('request_10', drop_salary)
show('request_10', 'no income pattern', a, 12700.0)

# also: user_10 payout stats
ev = data['events']
g = ev[(ev['user_id']=='user_10') & (ev['category']=='salary') & (ev['status']=='settled')]
print(f"    payouts: n={len(g)}, median={g['amount'].median():,.2f}, mean={g['amount'].mean():,.2f}, min={g['amount'].min():,.2f}, max={g['amount'].max():,.2f}")
print(f"    last 6:")
for _, r in g.tail(6).iterrows():
    print(f"      {r['settlement_date']} {r['amount']:>10,.2f} {r['description']}")

# ---- check: how many of the 21 are within +-25% (variance-explained) ----
