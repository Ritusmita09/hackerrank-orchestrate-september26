# AI Financial Agent - Core Engine

The engine is the deterministic, offline, LLM-free heart of the Financial Agent. 
It defines the canonical `FinancialState`, simulates cashflow against user constraints, and strictly verifies payment recommendations.

## Purpose and Rules

1. **Safety First**: The core ensures that no recommended payment plan will breach the user's `minimum_balance_to_keep`.
2. **Deterministic Verification**: Every recommendation is ultimately validated by `verifier.py`, which is independent of the generative logic (planner) and completely disconnected from any AI/LLM.
3. **Integer Arithmetic**: Floating-point math is banned here. Our `Money` model uses integers representing the smallest unit of currency (e.g., cents) to eliminate rounding errors.
4. **No Side Effects**: The forecast module does not mutate state. It returns a `CashflowSeries` representing simulated daily balances.

For deep architectural logic, refer to [ENGINE_DESIGN.md](./ENGINE_DESIGN.md).
