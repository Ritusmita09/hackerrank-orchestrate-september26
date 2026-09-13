"""P4 final: (1) which of the 25 samples have a pre-request failed debit;
(2) in-memory test: skip the FIRST future occurrence of a pattern whose category
had a failed debit shortly before request_date. Generalized rule, no IDs."""
import sys
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
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

print("=== 1. failed debits before request_date, per sample user ===")
for rid in sorted(samples, key=lambda r: int(r.split('_')[1])):
    s = samples[rid]
    rd = s['request_date']
    f = events[(events['user_id'] == s['user_id']) & (events['status'] == 'failed') &
               (events['direction'] == 'debit') & (events['settlement_date'] < rd)]
    if len(f):
        exp = float(s['amount_safe_to_pay'])
        print(f"  {rid} rd={rd.date()}: {len(f)} failed -> " +
              "; ".join(f"{r.settlement_date.date()} {r.category} {r.amount:,.0f}"
                        for r in f.itertuples()) + f"  (exp={exp:,.2f})")

# ---- in-memory rule test ----
SKIP_WINDOW = 35  # days before request_date that a failed debit suppresses next occurrence

orig_generate = fe.recurrence_detector.generate_future_occurrences

def patched_generate(pattern, start_date, days):
    occs = orig_generate(pattern, start_date, days)
    return occs  # placeholder; suppression applied below per-state

def build(state, suppress):
    sched = {}
    for event in state.scheduled_events:
        d = event['settlement_date']
        if state.as_of_date <= d <= state.as_of_date + timedelta(days=90):
            sched[d] = sched.get(d, 0) + (event['amount'] if event['direction'] == 'credit' else -event['amount'])
    sched_keys = {(e['settlement_date'], e['category'], e['direction']) for e in state.scheduled_events}
    flows = defaultdict(float)
    for d, a in sched.items():
        flows[d] += a
    for pattern in state.recurring_patterns:
        occs = orig_generate(pattern, state.as_of_date, 90)
        occs = [o for o in occs if (o['settlement_date'], o['category'], o['direction']) not in sched_keys]
        if suppress and pattern.category in suppress:
            occs = occs[1:]  # skip first occurrence
        for o in occs:
            flows[o['settlement_date']] += o['amount'] if o['direction'] == 'credit' else -o['amount']
    fc = {}
    bal = state.current_available_balance - sum(e['amount'] for e in state.pending_debits)
    d = state.as_of_date
    while d <= state.as_of_date + timedelta(days=90):
        bal += flows.get(d, 0)
        fc[d] = bal
        d += timedelta(days=1)
    return fc

def safe_amount(state, fc, requested):
    def ok(amount):
        return all(b - (amount if d >= state.as_of_date else 0) >= state.minimum_balance_to_keep - 1e-9
                   for d, b in fc.items())
    if ok(requested):
        return requested
    lo, hi = 0.0, requested
    while hi - lo > 0.01:
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return round(lo, 2)

print("\n=== 2. effect of 'skip first occurrence after recent failed debit' ===")
n_changed = 0
for rid in sorted(samples, key=lambda r: int(r.split('_')[1])):
    s = samples[rid]
    rd = s['request_date']
    state = fe.reconstruct_state(s['user_id'], rd, rid)
    fc0 = build(state, suppress=None)
    base = safe_amount(state, fc0, float(s['requested_amount']))

    f = events[(events['user_id'] == s['user_id']) & (events['status'] == 'failed') &
               (events['direction'] == 'debit') &
               (events['settlement_date'] < rd) &
               (events['settlement_date'] >= rd - timedelta(days=SKIP_WINDOW))]
    suppress = set(f['category']) if len(f) else None
    if not suppress:
        continue
    fc1 = build(state, suppress=suppress)
    new = safe_amount(state, fc1, float(s['requested_amount']))
    exp = float(s['amount_safe_to_pay'])
    mark = ' <-- RECONCILES' if abs(new - exp) <= 0.01 else ''
    print(f"  {rid}: base={base:,.2f} -> {new:,.2f} (exp={exp:,.2f}, suppress={suppress}){mark}")
    n_changed += 1
print(f"\nsamples affected: {n_changed}")
