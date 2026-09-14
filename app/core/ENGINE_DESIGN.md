# Engine Design: Architecture & Constraints

## LLM Isolation
The LLM is explicitly excluded from the financial verification phase. Large Language Models are prone to arithmetic hallucination, making them unsuitable for safety-critical budget simulations.
Instead, the LLM will be used strictly as an orchestration layer (later phases) to convert natural language into `FinancialState` (ingestion) and `AffordabilityRequest` objects. The outputs are generated locally, and the LLM merely structures the response for the user.

## Core Mechanies

### Forecasting (`forecast.py`)
A pure function `(FinancialState, Assumptions, SimulationScenario) -> CashflowSeries`.
It applies pending reservations, confirmed events, and un-deduplicated recurring patterns in strict chronological order. By tracking daily opening/closing balances against `minimum_balance_to_keep`, we observe breaches exactly when they occur.

### Affordability (`affordability.py`)
Instead of heuristic approximation, `maximum_safe_payment()` uses exact binary search over the integer space `[0, peak_available_limit]`. Because `forecast` is monotonic—a smaller payment is rigorously safer than a larger payment, everything else being equal—binary search operates deterministically without guessing.

### Planning (`planner.py`)
Generates multiple viable `PaymentPlan` candidates (e.g., Full Payment today, Wait 4 days, Installments, cuts to discretionary spending). This relies heavily on brute-forcing affordability simulations constraint sets (like combinations of allowed spending cuts). It acts purely as a proposer.

### Verification Gate (`verifier.py`)
Before *any* plan is emitted to the user, `verifier.py` simulates it from scratch:
- **Structural**: Does it contain correct dates, bounds, and currencies?
- **Constraints**: Are the obligations met? (Amounts & exact deadlines)
- **Permissions**: Are the spending cuts targeting protected areas? Did the user authorize installments?
- **Safety**: Does `forecast` drop below the minimum balance?

Verification ensures true LLM-safety. Even if an LLM is given access to a direct code-execution environment to "guess" a plan, it must pass `verify_plan`, which acts as an unassailable financial firewall.
