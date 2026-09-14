from .base import BaseTool
from .registry import ToolRegistry, registry
from .state_tool import StateTool
from .forecast_tool import ForecastTool
from .affordability_tool import AffordabilityTool
from .transaction_tool import TransactionTool
from .document_tools import ConfirmExtractionTool, ExtractDocumentTool
from .memory_tools import (
    GetPreferencesTool,
    ListGoalsTool,
    SaveFactTool,
    SaveGoalTool,
    SavePreferenceTool,
    SearchFactsTool,
)

# Auto-register all core tools
registry.register(StateTool)
registry.register(ForecastTool)
registry.register(AffordabilityTool)
registry.register(TransactionTool)

# Document tools: extraction is a proposal-only path - drafts are stored for
# the user to confirm; confirm_extraction is the only tool that moves a
# document into the ledger, and only on explicit user approval
# (ARCHITECTURE.md §5.1/§5.2). State-changing, never auto-invoked.
registry.register(ExtractDocumentTool)
registry.register(ConfirmExtractionTool)

# Memory tools: the agent's explicit, audited read/write path to user memory
# (ARCHITECTURE.md §4.2). State-changing, so they are never auto-invoked.
registry.register(GetPreferencesTool)
registry.register(SavePreferenceTool)
registry.register(ListGoalsTool)
registry.register(SaveGoalTool)
registry.register(SearchFactsTool)
registry.register(SaveFactTool)

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "registry",
    "StateTool",
    "ForecastTool",
    "AffordabilityTool",
    "TransactionTool",
    "ExtractDocumentTool",
    "ConfirmExtractionTool",
    "GetPreferencesTool",
    "SavePreferenceTool",
    "ListGoalsTool",
    "SaveGoalTool",
    "SearchFactsTool",
    "SaveFactTool",
]
