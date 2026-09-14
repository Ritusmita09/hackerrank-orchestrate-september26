"""LLM provider adapters (Phase 4).

A provider adapter implements the ``LLMProvider`` protocol consumed by
``AgentOrchestrator`` — ``generate(messages) -> LLMResponse`` — so a real
model backend can replace the canned test LLM without touching the
orchestrator, context assembly, or the tool registry.
"""
