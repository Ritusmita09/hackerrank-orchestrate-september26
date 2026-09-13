"""
Reverse-engineer the expected solution's forecast for all amount_safe_to_pay mismatches.

For each mismatched sample request:
  - implied_min_forecast = minimum_balance_to_keep + expected_safe  (if expected < requested)
  - our_min_forecast     = min of our 90-day forecast
  - gap = our_min - implied_min  (positive => we are too optimistic,
                                  negative => we are too pessimistic)
Also lists pending/scheduled events for the user around the request window.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'code'))

import pandas as pd
from data_loader import DataLoader
from financial_engine import FinancialEngine
from image_extractor import ImageExtractor

loader = DataLoader()
data = loader.load_all()
lookups = loader.build_lookup_structures(data)
fe = FinancialEngine(data, lookups, image_extractor=ImageExtractor())

samples = data['sample_requests'].to_dict('records')

rows = []
for s in samples:
    rid = s['request_id']
    state = fe.reconstruct_state(s['user_id'], s['request_date'], rid)
    forecast = fe.forecast_balance(state, days=90)
    ours = fe.calculate_safe_amount(state, s['requested_amount'], forecast)
    exp = float(s['amount_safe_to_pay'])
    if abs(ours - exp) < 0.01:
        continue  # match

    our_min = min(forecast.values())
    our_min_date = min(forecast, key=forecast.get)
    if exp < s['requested_amount']:
        implied_min = state.minimum_balance_to_keep + exp
        gap = our_min - implied_min
    else:
        implied_min = None
        gap = None

    # pending / scheduled events for this user (any date)
    uev = data['events'][data['events']['user_id'] == s['user_id']]
    non_settled = uev[uev['status'].isin(['pending', 'scheduled'])]
    ns_desc = '; '.join(
        f"{r.status}/{r.direction}/{r.category}/{r.amount}/{r.settlement_date.date()}"
        for r in non_settled.itertuples()
    ) or 'none'

    rows.append({
        'rid': rid,
        'user': s['user_id'],
        'req_date': s['request_date'].date(),
        'expected': exp,
        'ours': ours,
        'diff(ours-exp)': round(ours - exp, 2),
        'our_min_fc': round(our_min, 2),
        'our_min_date': str(our_min_date.date()),
        'implied_min': None if implied_min is None else round(implied_min, 2),
        'gap': None if gap is None else round(gap, 2),
        'non_settled_events': ns_desc,
    })

df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 60)
print(df.to_string(index=False))
print()
print('Too optimistic (ours > expected):', (df['diff(ours-exp)'] > 0).sum())
print('Too pessimistic (ours < expected):', (df['diff(ours-exp)'] < 0).sum())
