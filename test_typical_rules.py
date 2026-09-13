"""
Test candidate typical_amount rules against implied expected troughs.

implied_trough = minimum_balance_to_keep + expected_safe  (exact when expected < requested)
Rules: max (current), mean, median, min, last (most recent).
Also tests de-duplicating pattern occurrences that coincide with scheduled events.
"""
import sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'code'))

from data_loader import DataLoader
from financial_engine import FinancialEngine
from image_extractor import ImageExtractor

loader = DataLoader()
data = loader.load_all()
lookups = loader.build_lookup_structures(data)
fe = FinancialEngine(data, lookups, image_extractor=ImageExtractor())

samples = {s['request_id']: s for s in data['sample_requests'].to_dict('records')}

RULES = ['max', 'mean', 'median', 'min', 'last']

def rebuild_forecast(state, rule, dedup):
    # Recompute typical amounts per pattern from historical events
    hist = {e['event_id']: e for e in state.historical_events}
    patterns = []
    for p in state.recurring_patterns:
        amounts = [hist[eid]['amount'] for eid in p.event_ids if eid in hist
                   and hist[eid]['amount'] is not None]
        if not amounts:
            patterns.append(p)
            continue
        if rule == 'max':
            t = max(amounts)
        elif rule == 'mean':
            t = sum(amounts) / len(amounts)
        elif rule == 'median':
            t = statistics.median(amounts)
        elif rule == 'min':
            t = min(amounts)
        elif rule == 'last':
            # most recent = last in event_ids order (sorted by date at detection)
            t = amounts[-1]
        p2 = type(p)(p.category, p.period_days, t, p.last_date, p.event_ids,
                     p.flexibility, p.minimum_allowed_amount, p.day_of_month)
        patterns.append(p2)

    # Build forecast mirroring FinancialEngine.forecast_balance
    from collections import defaultdict
    from datetime import timedelta
    cash = defaultdict(float)
    start = state.as_of_date
    end = start + timedelta(days=90)
    sched_keys = {(e['settlement_date'], e['category'], e['direction']) for e in state.scheduled_events}
    for e in state.scheduled_events:
        if start <= e['settlement_date'] <= end:
            cash[e['settlement_date']] += e['amount'] if e['direction'] == 'credit' else -e['amount']
    for p in patterns:
        for o in fe.recurrence_detector.generate_future_occurrences(p, start, 90):
            d = o['settlement_date']
            if dedup and (d, p.category, o['direction']) in sched_keys:
                continue
            cash[d] += o['amount'] if o['direction'] == 'credit' else -o['amount']
    bal = state.current_available_balance
    for pd_ in state.pending_debits:
        bal -= pd_['amount']
    fc = {}
    d = start
    while d <= end:
        bal += cash.get(d, 0.0)
        fc[d] = bal
        d += timedelta(days=1)
    return fc

print(f"{'rid':<11}{'gap_now':>12}" + ''.join(f"{r:>12}" for r in RULES) + f"{'mean+dedup':>12}")
for rid, s in sorted(samples.items()):
    state = fe.reconstruct_state(s['user_id'], s['request_date'], rid)
    fc0 = fe.forecast_balance(state, days=90)
    ours = fe.calculate_safe_amount(state, s['requested_amount'], fc0)
    exp = float(s['amount_safe_to_pay'])
    if abs(ours - exp) < 0.01 or exp >= s['requested_amount']:
        continue
    implied = state.minimum_balance_to_keep + exp
    row = [round(min(fc0.values()) - implied, 2)]
    fcs = {}
    for rule in RULES + ['mean']:
        fcs[rule] = rebuild_forecast(state, rule, dedup=(rule == 'mean' and rule == 'mean' and False))
    # separate: mean without dedup already computed; now mean with dedup
    fc_md = rebuild_forecast(state, 'mean', dedup=True)
    for rule in RULES:
        row.append(round(min(fcs[rule].values()) - implied, 2))
    row.append(round(min(fc_md.values()) - implied, 2))
    print(f"{rid:<11}" + ''.join(f"{v:>12,.2f}" for v in row))
