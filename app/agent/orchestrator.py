import json
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol
from pydantic import BaseModel, Field

from sqlalchemy.orm import Session
from app.db.models import User
from app.agent.tools.registry import registry

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]

class Attachment(BaseModel):
    """One binary attachment (e.g. a document image) carried on a message.

    The orchestrator treats attachments as opaque: providers that support
    multimodal input translate them into their wire format; providers that do
    not simply never receive a message carrying them. ``data_b64`` is the
    base64-encoded payload, never raw bytes, so the model stays JSON-safe.
    """
    media_type: str   # e.g. "image/png"
    data_b64: str     # base64-encoded content


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    is_error: bool = False
    attachments: Optional[List[Attachment]] = None

class LLMResponse(BaseModel):
    text_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class LLMProvider(Protocol):
    def generate(self, messages: List[Message]) -> LLMResponse:
        """Generate the next response given a history of messages."""
        ...

class OrchestratorError(Exception):
    pass

class MaxStepsExceededError(OrchestratorError):
    pass

class AgentOrchestrator:
    """Independent orchestration loop coordinating LLMs and deterministic tools."""

    def __init__(
        self,
        db: Session,
        user: User,
        llm: LLMProvider,
        system_prompt: str = "",
        max_steps: int = 15
    ):
        self.db = db
        self.user = user
        self.llm = llm
        self.max_steps = max_steps
        self.messages: List[Message] = []
        if system_prompt:
            self.messages.append(Message(role=Role.SYSTEM, content=system_prompt))

    def run(self, user_prompt: str, as_of: date) -> str:
        """Run the orchestrator loop until completion or exception."""
        self.messages.append(Message(role=Role.USER, content=user_prompt))

        for step in range(self.max_steps):
            response = self.llm.generate(self.messages)

            # Store the assistant's response in history
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=response.text_content or "",
                tool_calls=response.tool_calls
            )
            self.messages.append(assistant_msg)

            # Check termination
            if not response.tool_calls:
                return response.text_content or ""

            # Dispatch tool calls in sequence
            for tool_call in response.tool_calls:
                try:
                    result_data = registry.execute(
                        tool_call.name,
                        self.db,
                        self.user,
                        tool_call.arguments,
                        as_of
                    )
                    import json
                    result_str = json.dumps(result_data, default=str)
                    self.messages.append(Message(
                        role=Role.TOOL,
                        tool_call_id=tool_call.id,
                        content=result_str,
                        is_error=False
                    ))
                except Exception as e:
                    # Capture formatting, validation, and missing-tool errors
                    # so the LLM can try to recover.
                    self.messages.append(Message(
                        role=Role.TOOL,
                        tool_call_id=tool_call.id,
                        content=f"Error: {type(e).__name__}: {str(e)}",
                        is_error=True
                    ))

        # Reached the end of the budget without a final non-tool answer
        raise MaxStepsExceededError(f"Agent exceeded maximum tool-call budget of {self.max_steps} steps.")

