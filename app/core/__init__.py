"""
Deterministic Financial Engine — the safety-critical core.

Architecture rule (non-negotiable):
    This package has ZERO LLM, network, database, or clock dependencies.
    It is fully testable offline. Time is always an input, never read.

See ENGINE_DESIGN.md at the repository root for the full design rationale.
"""
