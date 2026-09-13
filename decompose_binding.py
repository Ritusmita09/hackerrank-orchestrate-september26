"""Decompose the binding-date dip for each mismatched sample.

For each request: dump balance, pending debits, scheduled events in window,
recurring patterns, and the cash flows between request_date and binding_date.
Then compare (actual - expected) against those flows.
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "code"))
import pandas as pd
from collections import defaultdict

rows = pickle.load(open('_safe_amount_full.pkl', 'rb'))

for r in rows:
    rid, exp, act = r['rid'], r['exp'], r['actual']
    if abs(act - exp) < 0.01:
        continue
    state, forecast, rd = r['state'], r['forecast'], r['bind_date'] - pd.Timedelta(days=r['days'])
    rd = state.as_of_date
    bd = r['bind_date']
    delta = act - exp
    print(f"\n{'='*100}\n{rid}: expected={exp:,.2f} actual={act:,.2f} delta={delta:+,.2f} ({delta/exp*100:+.1f}%)")
    print(f"  balance={state.current_available_balance:,.2f} minbal={state.minimum_balance_to_keep:,.2f} "
          f"requested={r['req']:,.2f} binding={bd.date()} (day {r['days']})")
    pd_tot = sum(p['amount'] for p in state.pending_debits)
    if state.pending_debits:
        for p in state.pending_debits:
            print(f"    pending: {p['event_id']} {p['description'][:35]} {p['amount']:,.2f} {p['settlement_date'].date()}")
    else:
        print("    pending: none")
    # scheduled events in [rd, bd]
    sched = [e for e in state.scheduled_events if rd <= e['settlement_date'] <= bd]
    if sched:
        for e in sched:
            print(f"    sched:   {e['event_id']} {e['description'][:35]} {e['direction']} {e['amount']:,.2f} {e['settlement_date'].date()}")
    else:
        print("    sched:   none in window")
    # recurring patterns and their occurrences in [rd, bd]
    from recurrence import RecurrenceDetector  # noqa
    det = state.recurring_patterns
    for pat in det:
        occ = []
        d = rd
        # regenerate occurrences in window using engine's method
        pass
    print(f"    patterns: {len(det)}")
    for pat in det:
        nxt = pat.next_occurrence(rd)
        in_win = nxt <= bd
        print(f"      {pat.category:<18} typ={pat.typical_amount:>12,.2f} per={pat.period_days}d "
              f"dom={pat.days_of_month} flex={pat.flexibility[:4]} next={nxt.date()}{'  *BINDING-WINDOW*' if in_win else ''}")
        # occurrences within window
        d = nxt; n = 0; tot = 0.0
        while d <= bd and n < 10:
            tot += pat.typical_amount; n += 1
            d = pat.next_occurrence(d + pd.Timedelta(days=1))
        if n:
            print(f"        -> {n} occurrence(s) before binding, total typ {tot:,.2f}")
