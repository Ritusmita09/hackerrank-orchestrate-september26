from typing import Any, Dict, Type
from app.agent.tools.base import BaseTool

class ToolRegistry:
    """Registry holding all available LLM tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool_class: Type[BaseTool]) -> None:
        """Register a tool instance."""
        tool = tool_class()
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        """Retrieve a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool {name!r} not found in registry.")
        return self._tools[name]

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Get the JSON schemas for all registered tools."""
        return [tool.get_tool_schema() for tool in self._tools.values()]

    def execute(self, name: str, db, user, args: dict[str, Any], as_of) -> Any:
        tool = self.get_tool(name)
        # Validate arguments according to schema, then execute
        validated_args = tool.args_schema.model_validate(args).model_dump()
        return tool.execute(db, user, validated_args, as_of)

# Global registry instance
registry = ToolRegistry()
