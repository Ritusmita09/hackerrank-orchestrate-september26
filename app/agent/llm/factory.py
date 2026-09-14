"""Provider selection — the single switch between a canned and a real LLM.

``LLM_PROVIDER`` chooses the backend:

* ``none``      — no provider configured; callers that need one must supply
                  their own (the test suite injects canned transcripts).
* ``omniroute``  — the real adapter, talking to the local routing layer.

The orchestrator never sees this module: it works against the ``LLMProvider``
protocol, so swapping backends is configuration, not a code change
(PROJECT_SPEC §5.9).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from app.agent.orchestrator import LLMProvider
from app.agent.llm.omniroute_provider import OmniRouteProvider

#: Provider names that mean "no real backend configured".
_DISABLED_PROVIDERS = {"", "none"}


def create_llm_provider(
    settings: Optional[Any] = None,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> Optional[LLMProvider]:
    """Build the configured LLM provider, or ``None`` when none is configured.

    Args:
        settings: a ``Settings`` instance; the cached app settings are used
            when omitted.
        tools: tool schemas to advertise to the model. Defaults to the live
            registry — the registry stays the single source of truth for
            what tools exist. Pass ``[]`` for turn types that need no tools
            (e.g. structured extraction).
        model: model override for this turn type (ARCHITECTURE.md §4.3 model
            routing). When omitted the provider's default (``OMNIROUTE_MODEL``)
            applies.

    Raises:
        OmniRouteError: the configured provider is missing required
            configuration (e.g. no base URL or model).
        ValueError: ``LLM_PROVIDER`` names an unknown backend.
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings()

    name = (getattr(settings, "llm_provider", "") or "").strip().lower()

    if name in _DISABLED_PROVIDERS:
        return None

    if name == "omniroute":
        if tools is None:
            from app.agent.tools.registry import registry

            tools = registry.get_all_schemas()
        chosen_model = model or getattr(settings, "omniroute_model", None)
        return OmniRouteProvider(
            base_url=getattr(settings, "omniroute_base_url", None),
            api_key=getattr(settings, "omniroute_api_key", None),
            model=chosen_model,
            tools=tools,
            max_tokens=int(getattr(settings, "llm_max_tokens_per_turn", 4000) or 4000),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER {name!r}; expected one of: none, omniroute."
    )


def create_extraction_provider(settings: Optional[Any] = None) -> Optional[LLMProvider]:
    """Build the provider for extraction turns (multimodal, structured output).

    Extraction turns advertise no tools and may route to a dedicated multimodal
    model (``OMNIROUTE_EXTRACTION_MODEL``), falling back to the default model.
    Returns ``None`` when no LLM is configured — callers must surface that as
    a domain error, never fake an extraction.
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings()

    extraction_model = (getattr(settings, "omniroute_extraction_model", "") or "").strip()
    return create_llm_provider(settings, tools=[], model=extraction_model or None)
