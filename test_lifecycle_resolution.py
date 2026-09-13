"""Temporary lifecycle/duplicate-resolution investigation (NO production edits).

Hypothesis (user audit): the spec requires duplicate records to be ignored and
explicit cancellation/settlement/amendment to take precedence, using
linked_event_id to group transaction lifecycles. The production pipeline only
filters by status (cancelled/failed/unrealized dropped; pending debits reserved;
pending credits ignored) and dedups scheduled-vs-pattern occurrences inside
forecast_balance().

This script:
  1. Audits the lifecycle structure of financial_events.csv (link-based groups).
  2. Builds a generalizable resolver (link-based only, no request IDs, no
     expected values, no description keywords):
       - parent cancelled/failed            -> keep child only (no-op vs prod)
       - child unrealized                   -> no-op (prod already ignores)
       - child PENDING, same amount+direction as settled parent -> drop child
         (duplicate record; prod wrongly reserves it as a pending debit)
       - child SETTLED credit, same amount as settled debit parent -> drop BOTH
         (explicit settlement: reversal/reimbursement cancels the charge; the
         debit must not count as historical consumption)
       - child pending credit (refund in flight) -> keep parent (not settled yet)
  3. Runs the FULL real pipeline (process_request) on all 25 labeled samples
     twice: production events vs resolved events; reports amount/full matches,
     abs/signed error, changed requests, exact-four integrity.
  4. Checks whether pattern statistics currently include lifecycle events.
"""
import sys
import io
import copy
import contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'code'))
import pandas as pd
from data_loader import DataLoader
from image_extractor import ImageExtractor
from financial_engine import FinancialEngine
from decision_engine import DecisionEngine
from main import process_request

loader = DataLoader()
data = loader.load_all()

SAMPLES = data['sample_requests'].to_dict('records')
FIELDS = ['amount_safe_to_pay', 'affordability_status', 'recommended_payment_method',
          'payment_plan', 'earliest_date_for_full_payment', 'spending_changes_needed']


def compare_field(actual, expected, field_name):
    actual_str = str(actual).strip() if pd.notna(actual) else ''
    expected_str = str(expected).strip() if pd.notna(expected) else ''
    if actual_str == expected_str:
        return True
    if field_name == 'earliest_date_for_full_payment':
        try:
            if actual_str and expected_str:
                if pd.to_datetime(actual_str).date() == pd.to_datetime(expected_str).date():
                    return True
        except Exception:
            pass
    if field_name == 'amount_safe_to_pay':
        try:
            if abs(float(actual_str) - float(expected_str)) < 0.01:
                return True
        except Exception:
            pass
    return False


# ---------------- 1. Audit lifecycle groups ----------------
ev = data['events'].copy()
by_id = {r['event_id']: r for r in ev.to_dict('records')}

groups = {}   # parent_id -> {'parent': row, 'children': [rows]}
for r in ev.to_dict('records'):
    lid = r['linked_event_id']
    if isinstance(lid, str) and lid.strip():
        groups.setdefault(lid, {'parent': by_id.get(lid), 'children': []})['children'].append(r)

print(f"Lifecycle link groups: {len(groups)} (children linked to {len(groups)} distinct parents)")

archetype_counts = {}
for pid, g in groups.items():
    p, kids = g['parent'], g['children']
    for c in kids:
        if p is None or p['status'] in ('cancelled', 'failed'):
            key = f"parent {p['status'] if p else 'missing'} -> child {c['status']} (no-op: prod ignores parent)"
        elif c['status'] == 'unrealized':
            key = "parent settled -> child unrealized (no-op: prod ignores unrealized)"
        elif c['status'] == 'pending' and c['direction'] == p['direction'] and c['amount'] == p['amount']:
            key = "DUPLICATE: settled parent + pending same-amount child (prod RESERVES the duplicate)"
        elif c['status'] == 'settled' and c['direction'] == 'credit' and p['direction'] == 'debit' and c['amount'] == p['amount']:
            key = "REVERSED/REIMBURSED: settled debit + settled credit same amount (prod keeps BOTH in history)"
        elif c['status'] == 'pending' and c['direction'] == 'credit':
            key = "REFUND IN FLIGHT: settled debit + pending credit (correct: prod keeps debit, ignores credit)"
        else:
            key = f"other: parent {p['status']} {p['direction']} -> child {c['status']} {c['direction']}"
        archetype_counts[key] = archetype_counts.get(key, 0) + 1

print("\nArchetypes:")
for k, v in sorted(archetype_counts.items(), key=lambda x: -x[1]):
    print(f"  {v:3d}  {k}")

sample_users = set(s['user_id'] for s in SAMPLES)
affected_users = {pid: [r['user_id'] for r in g['children']] for pid, g in groups.items()}
sample_hit = {pid: u for pid, u in affected_users.items() if any(x in sample_users for x in u)}
print(f"\nGroups touching the 25 sample users: {len(sample_hit)} -> {sorted(sample_hit)}")

# ---------------- 2. Build resolver ----------------
def resolve_events(df):
    """Return (resolved_df, dropped_ids, reason_by_id). Link-based, no keywords."""
    recs = df.to_dict('records')
    by = {r['event_id']: r for r in recs}
    dropped, reason = set(), {}
    for r in recs:
        lid = r['linked_event_id']
        if not (isinstance(lid, str) and lid.strip()):
            continue
        p = by.get(lid)
        if p is None:
            continue
        # duplicate: pending child mirroring a settled parent (same amount+direction)
        if (r['status'] == 'pending' and p['status'] == 'settled'
                and r['direction'] == p['direction'] and r['amount'] == p['amount']):
            dropped.add(r['event_id'])
            reason[r['event_id']] = 'duplicate pending record'
        # explicit settlement: settled credit cancelling a settled debit (same amount)
        elif (r['status'] == 'settled' and p['status'] == 'settled'
              and r['direction'] == 'credit' and p['direction'] == 'debit'
              and r['amount'] == p['amount']):
            dropped.add(p['event_id']); reason[p['event_id']] = 'reversed/reimbursed charge'
            dropped.add(r['event_id']); reason[r['event_id']] = 'reversal/reimbursement credit'
    kept = df[~df['event_id'].isin(dropped)].copy()
    return kept, dropped, reason


resolved_df, dropped_ids, reason = resolve_events(ev)
print(f"\nResolver drops {len(dropped_ids)} events "
      f"({sum(1 for v in reason.values() if 'duplicate' in v)} duplicate children, "
      f"{sum(1 for v in reason.values() if 'charge' in v)} reversed charges, "
      f"{sum(1 for v in reason.values() if 'credit' in v)} reversal credits)")

sample_dropped = [(i, reason[i], by_id[i]['user_id'], by_id[i]['description'],
                   by_id[i]['amount'], by_id[i]['status'])
                  for i in sorted(dropped_ids) if by_id[i]['user_id'] in sample_users]
print(f"Dropped events belonging to sample users: {len(sample_dropped)}")
for x in sample_dropped:
    print(f"  {x[0]} ({x[1]}): user {x[2]} {x[3]} {x[4]} {x[5]}")


# ---------------- 3. Run pipeline both ways ----------------
def run_pipeline(events_df):
    d = dict(data)  # shallow copy; replace events
    d['financial_events'] = events_df
    lk = loader.build_lookup_structures(d)
    fe = FinancialEngine(d, lk, ImageExtractor())
    de = DecisionEngine(fe)
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for s in SAMPLES:
            o = process_request(s, d, lk, fe, de)
            full = all(compare_field(o[f], s[f], f) for f in FIELDS)
            out[s['request_id']] = {'amount': float(o['amount_safe_to_pay']), 'full': full}
    return out


print("\nRunning production pipeline (all 25)...")
base = run_pipeline(ev)
print("Running resolved-event pipeline (all 25)...")
res = run_pipeline(resolved_df)

EXPECTED = {s['request_id']: float(s['amount_safe_to_pay']) for s in SAMPLES}
EXACT_FOUR = [r for r in sorted(base) if abs(base[r]['amount'] - EXPECTED[r]) < 0.01]


def metrics(run):
    amt = sum(1 for r in run if abs(run[r]['amount'] - EXPECTED[r]) < 0.01)
    full = sum(1 for r in run if run[r]['full'])
    abs_err = sum(abs(run[r]['amount'] - EXPECTED[r]) for r in run)
    sgn_err = sum(run[r]['amount'] - EXPECTED[r] for r in run)
    return amt, full, abs_err, sgn_err


b_amt, b_full, b_abs, b_sgn = metrics(base)
r_amt, r_full, r_abs, r_sgn = metrics(res)
print(f"\n=== BEFORE (production) ===")
print(f"  amount matches: {b_amt}/25 | full matches: {b_full}/25 | abs err: {b_abs:,.2f} | signed err: {b_sgn:+,.2f}")
print(f"  exact requests: {EXACT_FOUR}")
print(f"\n=== AFTER (lifecycle-resolved) ===")
print(f"  amount matches: {r_amt}/25 | full matches: {r_full}/25 | abs err: {r_abs:,.2f} | signed err: {r_sgn:+,.2f}")

changed = [r for r in base if abs(base[r]['amount'] - res[r]['amount']) > 0.005]
print(f"\nChanged requests ({len(changed)}):")
for r in sorted(changed):
    eb = abs(base[r]['amount'] - EXPECTED[r])
    en = abs(res[r]['amount'] - EXPECTED[r])
    tag = 'IMPROVED' if en < eb - 0.005 else ('REGRESSED' if en > eb + 0.005 else 'same error')
    print(f"  {r}: {base[r]['amount']:,.2f} -> {res[r]['amount']:,.2f} (exp {EXPECTED[r]:,.2f}) [{tag}]")

broken = [r for r in EXACT_FOUR if abs(res[r]['amount'] - EXPECTED[r]) >= 0.01 or not res[r]['full']]
print(f"Exact matches broken: {broken if broken else 'NONE'}")


# ---------------- 4. Pattern-statistics pollution check ----------------
print("\n=== Pattern-statistics check: are lifecycle events inside pattern history? ===")
for s in SAMPLES:
    uid, rd = s['user_id'], pd.Timestamp(s['request_date'])
    lk = loader.build_lookup_structures(data)
    fe = FinancialEngine(data, lk, ImageExtractor())
    st = fe.reconstruct_state(uid, rd, s['request_id'])
    ids = {e['event_id'] for e in st.historical_events}
    poll = ids & dropped_ids
    pids = {e['event_id'] for p in st.pending_debits for e in [p]}
    ppoll = pids & dropped_ids
    if poll or ppoll:
        pats = {p.category: [round(p.typical_amount, 2)] for p in st.recurring_patterns}
        print(f"  {s['request_id']} (user {uid}): history-polluted={sorted(poll)} pending-polluted={sorted(ppoll)}")
