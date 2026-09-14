"""OmniRoute LLM provider adapter (Phase 4).

OmniRoute (a local model-routing layer) exposes the **Anthropic Messages API**
on its local port — the same endpoint form Claude Code itself uses via
``ANTHROPIC_BASE_URL``. This adapter talks that API and implements the
``LLMProvider`` protocol the orchestrator already consumes, so a real model
backend replaces the canned test LLM without a single change to
``AgentOrchestrator``, ``ContextBuilder``, the tool registry, or the engine.

Translation responsibilities (this module is the only place provider wire
format exists):

    internal Message list  ->  Anthropic request
      SYSTEM     -> top-level ``system`` string (Anthropic has no system role)
      USER       -> ``{"role": "user", "content": [...]}``
                    (``attachments`` become base64 ``image`` blocks first)
      ASSISTANT  -> ``{"role": "assistant", "content": [text?, tool_use...]}``
      TOOL       -> ``{"role": "user", "content": [tool_result...]}``
                    (consecutive results merge into one user turn, which is
                    what the API requires after a parallel tool_use turn)

    Anthropic response     ->  LLMResponse
      ``text`` blocks      -> ``text_content``
      ``tool_use`` blocks  -> ``ToolCall`` (id, name, input)
      other block types    -> ignored (e.g. ``thinking``)

Configuration is environment-only (see ``app/config.py``); no credential is
ever hardcoded, logged, or echoed in an error message.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import httpx

from app.agent.orchestrator import (
    LLMProvider,
    LLMResponse,
    Message,
    Role,
    ToolCall,
)

#: Anthropic API version header for the Messages API.
ANTHROPIC_VERSION = "2023-06-01"

#: Default request timeout (seconds) — the spec requires LLM calls to be bounded.
DEFAULT_TIMEOUT = 60.0

#: Placeholder substituted for any credential that would otherwise leak into
#: an error message, log line, or test failure output.
_REDACTED = "***redacted***"


class OmniRouteError(RuntimeError):
    """Any failure talking to the routing layer (config, transport, or payload)."""


class OmniRouteProvider(LLMProvider):
    """LLM provider backed by a local OmniRoute instance.

    ``generate`` is the only method the orchestrator calls; everything else is
    translation between the internal message model and the wire format.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        max_tokens: int = 4000,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        """Build a provider.

        Every credential-bearing argument falls back to its environment
        variable, so constructing from config is ``OmniRouteProvider()``.

        Args:
            base_url: OmniRoute root, e.g. ``http://localhost:20128``
                (env ``OMNIROUTE_BASE_URL``).
            api_key: OmniRoute access token (env ``OMNIROUTE_API_KEY``).
            model: provider-prefixed model id, e.g. ``agentrouter/deepseek-v4-flash``
                (env ``OMNIROUTE_MODEL``).
            tools: tool schemas in Anthropic form — exactly what
                ``registry.get_all_schemas()`` already returns.
            max_tokens: response cap for each call.
            timeout: HTTP timeout in seconds.
            transport: injected ``httpx`` transport; tests pass a
                ``MockTransport`` so no socket is ever opened.
        """
        self.base_url = (base_url if base_url is not None else os.environ.get("OMNIROUTE_BASE_URL", "")).rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get("OMNIROUTE_API_KEY", "")
        self.model = model if model is not None else os.environ.get("OMNIROUTE_MODEL", "")
        self.tools = list(tools) if tools else []
        self.max_tokens = max_tokens
        self.timeout = timeout

        if not self.base_url:
            raise OmniRouteError(
                "OmniRoute base URL is not configured (set OMNIROUTE_BASE_URL)."
            )
        if not self.model:
            raise OmniRouteError(
                "OmniRoute model is not configured (set OMNIROUTE_MODEL, "
                "provider-prefixed, e.g. 'agentrouter/deepseek-v4-flash')."
            )

        # Only pay for a client when we have one; a supplied transport keeps
        # tests fully offline.
        client_kwargs: Dict[str, Any] = {"timeout": timeout}
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.Client(**client_kwargs)

    # -- LLMProvider protocol -------------------------------------------------

    def generate(self, messages: List[Message]) -> LLMResponse:
        """Send the conversation to OmniRoute and translate the reply back."""
        payload = self._build_payload(messages)
        data = self._post(payload)
        return self._parse_response(data)

    # -- request construction -------------------------------------------------

    def _build_payload(self, messages: List[Message]) -> Dict[str, Any]:
        """Translate internal messages into an Anthropic Messages request."""
        system_parts: List[str] = []
        wire_messages: List[Dict[str, Any]] = []

        for message in messages:
            if message.role == Role.SYSTEM:
                if message.content:
                    system_parts.append(message.content)
                continue

            if message.role == Role.USER:
                # Multimodal turns: image blocks first (the API expects content
                # describing an image to precede the text that references it).
                for attachment in message.attachments or []:
                    self._append_block(
                        wire_messages,
                        "user",
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": attachment.media_type,
                                "data": attachment.data_b64,
                            },
                        },
                    )
                self._append_block(
                    wire_messages, "user", {"type": "text", "text": message.content}
                )
                continue

            if message.role == Role.ASSISTANT:
                blocks: List[Dict[str, Any]] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                for call in message.tool_calls or []:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                # An assistant turn with neither prose nor tool calls carries
                # nothing the API accepts; skip it rather than send empty content.
                if blocks:
                    wire_messages.append({"role": "assistant", "content": blocks})
                continue

            if message.role == Role.TOOL:
                self._append_tool_result(wire_messages, message)
                continue

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": wire_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if self.tools:
            payload["tools"] = self.tools
        return payload

    @staticmethod
    def _append_block(
        wire_messages: List[Dict[str, Any]], role: str, block: Dict[str, Any]
    ) -> None:
        """Append a content block, extending the same-role turn when possible.

        The API requires alternating user/assistant turns, so merging keeps
        consecutive same-role messages valid.
        """
        if wire_messages and wire_messages[-1]["role"] == role:
            wire_messages[-1]["content"].append(block)
        else:
            wire_messages.append({"role": role, "content": [block]})

    def _append_tool_result(
        self, wire_messages: List[Dict[str, Any]], message: Message
    ) -> None:
        """Translate a TOOL message into a ``tool_result`` content block.

        Tool results belong to *user* turns in this API, and several results
        from one parallel tool-call turn merge into a single user message.
        """
        if not message.tool_call_id:
            # Without a call id there is nothing to correlate against; keep the
            # information as plain text rather than dropping it silently.
            self._append_block(
                wire_messages, "user", {"type": "text", "text": message.content}
            )
            return

        block: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id,
            "content": message.content,
        }
        if message.is_error:
            # Surfaces the failure to the model as a failed tool result, which
            # is what lets it recover instead of treating output as success.
            block["is_error"] = True

        if wire_messages and wire_messages[-1]["role"] == "user" and all(
            b.get("type") == "tool_result" for b in wire_messages[-1]["content"]
        ):
            wire_messages[-1]["content"].append(block)
        else:
            wire_messages.append({"role": "user", "content": [block]})

    # -- transport ------------------------------------------------------------

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to the Messages endpoint, mapping every failure to OmniRouteError."""
        url = f"{self.base_url}/v1/messages"
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key

        try:
            response = self._client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            # Connection refused / DNS / timeout: report the cause, not the request.
            raise OmniRouteError(
                f"Could not reach OmniRoute at {self.base_url}: {self._scrub(str(exc))}"
            ) from None

        if response.status_code >= 400:
            raise OmniRouteError(self._http_error_message(response))

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise OmniRouteError(
                f"OmniRoute returned a non-JSON response: {self._scrub(str(exc))}"
            ) from None

        if not isinstance(data, dict):
            raise OmniRouteError("OmniRoute returned a malformed response (expected a JSON object).")
        return data

    def _http_error_message(self, response: httpx.Response) -> str:
        """Describe an HTTP failure, preferring the provider's own message."""
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message", ""))
                elif isinstance(error, str):
                    detail = error
                elif body.get("message"):
                    detail = str(body["message"])
        except (json.JSONDecodeError, ValueError):
            # Non-JSON error body: keep it short and never echo raw headers.
            detail = (response.text or "")[:200]

        message = f"OmniRoute API error (HTTP {response.status_code})"
        if detail:
            message = f"{message}: {self._scrub(detail)}"
        return message

    # -- response parsing -----------------------------------------------------

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Translate an Anthropic Messages response into an LLMResponse."""
        # An error payload can arrive with a 200 from some gateways.
        if isinstance(data.get("error"), (dict, str)):
            raise OmniRouteError(f"OmniRoute returned an error payload: {self._scrub(self._error_detail(data['error']))}")

        content = data.get("content")
        if content is None:
            raise OmniRouteError("OmniRoute response contained no 'content' field.")

        # Some providers emit a plain string instead of content blocks.
        if isinstance(content, str):
            return LLMResponse(text_content=content or None, tool_calls=None)

        if not isinstance(content, list):
            raise OmniRouteError(
                f"OmniRoute response 'content' was {type(content).__name__}, expected a list of blocks."
            )

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)

            elif block_type == "tool_use":
                name = block.get("name")
                if not name:
                    # A tool call we cannot name is not dispatchable; ignore it
                    # rather than fabricate a tool the registry will reject.
                    continue
                arguments = block.get("input")
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id") or ""),
                        name=str(name),
                        arguments=arguments if isinstance(arguments, dict) else {},
                    )
                )

            # Every other block type (e.g. "thinking") is intentionally ignored:
            # it is provider reasoning, not a message for the orchestrator.

        return LLMResponse(
            text_content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
        )

    @staticmethod
    def _error_detail(error: Any) -> str:
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or "unknown error")
        return str(error)

    # -- secret hygiene -------------------------------------------------------

    def _scrub(self, text: str) -> str:
        """Remove any credential from text bound for an exception or log."""
        if not text:
            return text
        scrubbed = text
        if self._api_key:
            scrubbed = scrubbed.replace(self._api_key, _REDACTED)
        return scrubbed

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "OmniRouteProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
