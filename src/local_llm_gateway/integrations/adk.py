"""Google ADK model adapter. ADK retains tool execution and conversation state."""

import copy
import json

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.models._capabilities import LlmCapabilities
from google.genai import types
from pydantic import BaseModel, Field

from ..contracts import GenerationRequest, Message, Tool, ToolCall
from ..errors import GatewayError
from ..gateway import Gateway


def json_schema(value):
    if isinstance(value, type) and issubclass(value, BaseModel):
        return value.model_json_schema()
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if not isinstance(value, dict):
        return value
    # Only descend through schema-bearing keywords. Property-name maps and
    # defaults/enum values are not schemas themselves.
    converted = copy.deepcopy(value)
    converted.pop("propertyOrdering", None)
    converted.pop("nullable", None)
    for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        if isinstance(value.get(key), dict):
            converted[key] = {name: json_schema(schema) for name, schema in value[key].items()}
    for key in (
        "items",
        "additionalProperties",
        "additionalItems",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
    ):
        if isinstance(value.get(key), dict):
            converted[key] = json_schema(value[key])
    for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
        if isinstance(value.get(key), list):
            converted[key] = [json_schema(schema) for schema in value[key]]
    if isinstance(converted.get("type"), str):
        converted["type"] = converted["type"].lower()
    if value.get("nullable"):
        return {"anyOf": [converted, {"type": "null"}]}
    return converted


def content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(content_text(v) for v in value)
    if isinstance(value, types.Content):
        return "\n".join(content_text(part) for part in value.parts or [])
    if isinstance(value, types.Part) and value.text is not None:
        return value.text
    raise GatewayError("unsupported_capability", "Only text system instructions are supported.")


class GatewayLlm(BaseLlm):
    model: str = "local-gateway"
    gateway: Gateway = Field(exclude=True)
    profile: str | None = None

    @classmethod
    def from_config(cls, path="llm-gateway.yaml", *, profile=None):
        return cls(gateway=Gateway.from_config(path), profile=profile)

    @property
    def capabilities(self):
        return LlmCapabilities(output_schema_and_tools=True)

    def _request(self, request: LlmRequest):
        config = request.config or types.GenerateContentConfig()
        # ADK adds labels for tracing; they are not model sampling controls.
        supported = {
            "system_instruction",
            "tools",
            "tool_config",
            "response_schema",
            "response_json_schema",
            "response_mime_type",
            "automatic_function_calling",
            "labels",
        }
        set_options = set(config.model_dump(exclude_none=True))
        unsupported = set_options - supported
        if unsupported:
            raise GatewayError(
                "unsupported_capability",
                "Unsupported ADK model options: " + ", ".join(sorted(unsupported)),
            )
        if config.response_mime_type not in {None, "text/plain", "application/json"}:
            raise GatewayError(
                "unsupported_capability", "Only text and JSON responses are supported."
            )
        messages = []
        if config.system_instruction:
            messages.append(Message(role="system", content=content_text(config.system_instruction)))
        pending: list[ToolCall] = []
        for content in request.contents:
            text, calls, results = [], [], []
            for part in content.parts or []:
                if part.thought:
                    continue
                if part.function_call:
                    call = part.function_call
                    converted = ToolCall(
                        name=call.name,
                        arguments=call.args or {},
                        **({"id": call.id} if call.id else {}),
                    )
                    calls.append(converted)
                    pending.append(converted)
                elif part.function_response:
                    result = part.function_response
                    matched = next(
                        (
                            c
                            for c in pending
                            if (c.id == result.id if result.id else c.name == result.name)
                        ),
                        None,
                    )
                    if matched is None:
                        raise GatewayError(
                            "invalid_request", "ADK tool response has no matching call."
                        )
                    pending.remove(matched)
                    results.append(
                        Message(
                            role="tool",
                            content=json.dumps(result.response),
                            tool_call_id=matched.id,
                            name=result.name,
                        )
                    )
                elif part.text is not None:
                    text.append(part.text)
                else:
                    raise GatewayError(
                        "unsupported_capability",
                        "Gateway v0.1 accepts text and function parts only.",
                    )
            if results:
                messages.extend(results)
            if text or calls:
                if content.role not in {"model", "user"}:
                    raise GatewayError("invalid_request", "Unsupported ADK content role.")
                messages.append(
                    Message(
                        role="assistant" if content.role == "model" else "user",
                        content="\n".join(text),
                        tool_calls=calls,
                    )
                )
        tools = []
        for group in config.tools or []:
            if not isinstance(group, types.Tool) or not group.function_declarations:
                raise GatewayError(
                    "unsupported_capability",
                    "Only ADK function tools are supported; provider-hosted tools are unavailable.",
                )
            other = set(group.model_dump(exclude_none=True)) - {"function_declarations"}
            if other:
                raise GatewayError(
                    "unsupported_capability", "Provider-hosted tool fields are unsupported."
                )
            for function in group.function_declarations:
                parameters = function.parameters_json_schema
                if parameters is None:
                    parameters = (
                        json_schema(function.parameters)
                        if function.parameters
                        else {"type": "object"}
                    )
                tools.append(
                    Tool(
                        name=function.name,
                        description=function.description or "",
                        parameters=parameters,
                    )
                )
        choice = "auto"
        if config.tool_config and config.tool_config.function_calling_config:
            fc = config.tool_config.function_calling_config
            if fc.mode == types.FunctionCallingConfigMode.NONE:
                choice = "none"
            elif fc.mode == types.FunctionCallingConfigMode.ANY:
                choice = "required"
            elif fc.mode not in {None, types.FunctionCallingConfigMode.AUTO}:
                raise GatewayError(
                    "unsupported_capability", "Unsupported ADK function-calling mode."
                )
            if fc.allowed_function_names:
                allowed = set(fc.allowed_function_names)
                if not allowed.issubset({t.name for t in tools}):
                    raise GatewayError("invalid_request", "Unknown ADK allowed tool name.")
                tools = [t for t in tools if t.name in allowed]
        schema = config.response_json_schema
        if schema is None and config.response_schema is not None:
            schema = json_schema(config.response_schema)
        if schema is None and config.response_mime_type == "application/json":
            schema = {}
        return GenerationRequest(
            messages=messages, tools=tools, tool_choice=choice, output_schema=schema
        )

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        if stream:
            raise GatewayError(
                "unsupported_capability",
                "Incremental ADK streaming is not supported; use non-streaming mode.",
            )
        response = await self.gateway.generate(self._request(llm_request), profile=self.profile)
        parts = []
        if response.text:
            parts.append(types.Part(text=response.text))
        parts.extend(
            types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.arguments))
            for c in response.tool_calls
        )
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
            model_version=response.reported_model,
            custom_metadata={
                "gateway_provider": response.provider,
                "gateway_model": response.model,
                "gateway_request_id": response.id,
                "gateway_usage": response.usage,
            },
        )
