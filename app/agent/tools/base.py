import abc
import datetime
from datetime import date
from typing import Any, ClassVar, Type
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db.models import User

class BaseTool(abc.ABC):
    """Abstract base class for all LLM tools."""

    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[Type[BaseModel]]

    @abc.abstractmethod
    def execute(self, db: Session, user: User, args: dict[str, Any], as_of: date) -> Any:
        """Execute the tool logic using the provided dependencies and arguments."""
        pass

    @classmethod
    def get_tool_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema describing this tool for the LLM."""
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": cls.args_schema.model_json_schema(),
        }
