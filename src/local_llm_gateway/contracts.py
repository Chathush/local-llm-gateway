from __future__ import annotations

import json
from typing import Any, Literal
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def checked_schema(schema: dict) -> dict:
    """Only local JSON Schema references; validation must never fetch URLs."""

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef"} and not (
                    isinstance(item, str) and item.startswith("#")
                ):
                    raise ValueError("Only local JSON Schema references are supported")
                if key == "$id":
                    raise ValueError("Schema $id is unsupported; use local $defs")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ValueError("Invalid JSON Schema") from exc
    return schema


class Tool(StrictModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})

    @field_validator("parameters")
    @classmethod
    def valid_parameters(cls, value):
        checked_schema(value)
        if value.get("type") != "object":
            raise ValueError("Tool parameters must be an object schema")
        return value


class ToolCall(StrictModel):
    id: str = Field(default_factory=lambda: "call_" + uuid4().hex, min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def validate_role(self):
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages may contain tool calls")
        if (self.role == "tool") != bool(self.tool_call_id):
            raise ValueError("Tool results require tool_call_id; other roles must omit it")
        return self


class GenerationRequest(StrictModel):
    messages: list[Message] = Field(min_length=1)
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: str = "auto"
    output_schema: dict[str, Any] | None = None

    @field_validator("output_schema")
    @classmethod
    def validate_schema(cls, value):
        return checked_schema(value) if value is not None else None

    @model_validator(mode="after")
    def validate_conversation(self):
        names = [t.name for t in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("Tool names must be unique")
        if self.tool_choice not in {"auto", "none", "required", *names}:
            raise ValueError("tool_choice must be auto, none, required, or an available tool name")
        if self.tool_choice == "required" and not self.tools:
            raise ValueError("required tool choice needs tools")
        seen, pending = set(), set()
        for message in self.messages:
            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("Tool result has no matching unanswered tool call")
                pending.remove(message.tool_call_id)
            elif pending:
                raise ValueError("Return all pending tool results before the next message")
            for call in message.tool_calls:
                if call.id in seen:
                    raise ValueError("Tool call IDs must be unique")
                seen.add(call.id)
                pending.add(call.id)
        if pending:
            raise ValueError("Conversation ends with unanswered tool calls")
        return self


class GenerationResponse(StrictModel):
    id: str = Field(default_factory=lambda: "gen_" + uuid4().hex)
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: Literal["stop", "tool_calls"]
    provider: str
    model: str | None
    reported_model: str | None = None
    usage: dict[str, Any] | None = None
    duration_seconds: float

    def as_message(self) -> Message:
        return Message(role="assistant", content=self.text, tool_calls=self.tool_calls)

    def json_output(self) -> Any:
        return json.loads(self.text)
