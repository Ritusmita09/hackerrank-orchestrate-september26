"""P4: our engine's patterns+forecast for user_05 (r05) and user_25 (r25);
also: does 'Final ... payroll' / failed-debit semantics show up elsewhere?"""
import sys
from pathlib import Path
from datetime import timedelta
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "code"))

from data_loader import DataLoader
from image_extractor import ImageExtractor
from financial_engine import FinancialEngine

loader = DataLoader()
data = loader.load_all()
lookups = loader.build_lookup_structures(data)
fe = FinancialEngine(data, lookups, ImageExtractor())
samples = {s['request_id']: s for s in data['sample_requests'].to_dict('records')}
events = data['events']

for rid in ['request_05', 'request_25']:
    s = samples[rid]
    state = fe.reconstruct_state(s['user_id'], s['request_date'], rid)
    print(f"===== {rid} ({s['user_id']}) rd={s['request_date'].date()} "
          f"ours vs exp =====")
    print(f"  bal={state.current_available_balance:,.2f} min_keep={state.minimum_balance_to_keep:,.2f} "
          f"pending_debits={sum(e['amount'] for e in state.pending_debits):,.2f}")
    print(f"  patterns detected:")
    for p in state.recurring_patterns:
        per = 'monthly' if p.period_days == 0 else f"{p.period_days}d"
        amts = [e['amount'] for e in state.historical_events if e['event_id'] in p.event_ids]
        print(f"    {p.category:<14} {per:<8} typical={p.typical_amount:>14,.2f} n_hist={len(amts)} amts={amts[:6]}")
    print(f"  scheduled events in window:")
    for e in state.scheduled_events:
        if state.as_of_date <= e['settlement_date'] <= state.as_of_date + timedelta(days=90):
            print(f"    {e['settlement_date'].date()} {e['direction']:<6} {e['amount']:>14,.2f} {e['category']}")
    # now the actual safe amount
    fc = fe.forecast_balance(state, 90)
    res = fe.calculate_safe_amount(state, float(s['requested_amount']), fc)
    print(f"  OURS={res:,.2f}  EXP={float(s['amount_safe_to_pay']):,.2f}  "
          f"err={res - float(s['amount_safe_to_pay']):+,.2f}")

print("\n=== 'Final' payroll/final-pay descriptions across full events ===")
mask = events['description'].str.contains('final', case=False, na=False)
print(events[mask][['event_id', 'user_id', 'settlement_date', 'status', 'direction',
                    'amount', 'category', 'description']].to_string(index=False))

print("\n=== failed debits: how many, and near-term retry patterns ===")
failed = events[(events['status'] == 'failed') & (events['direction'] == 'debit')]
print(f"total failed debits: {len(failed)}")
print(failed['category'].value_counts().to_string())
# do users with a failed debit later have a settled retry of same category?
retried = 0
for uid, grp in failed.groupby('user_id'):
    for _, f in grp.iterrows():
        later = events[(events['user_id'] == uid) & (events['category'] == f['category']) &
                       (events['direction'] == 'debit') & (events['status'] == 'settled') &
                       (events['settlement_date'] > f['settlement_date']) &
                       (events['settlement_date'] <= f['settlement_date'] + timedelta(days=30))]
        if len(later):
            retried += 1
print(f"failed debits with a settled same-category debit within 30d after: {retried}/{len(failed)}")
