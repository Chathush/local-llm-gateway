"""LangChain chat model, usable directly in LangGraph nodes and tool loops."""

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from ..contracts import GenerationRequest, Message, Tool, ToolCall
from ..errors import GatewayError
from ..gateway import Gateway


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
            else:
                raise GatewayError(
                    "unsupported_capability", "Gateway v0.1 accepts text content only."
                )
        return "\n".join(parts)
    raise GatewayError("unsupported_capability", "Unsupported message content.")


def tool_choice_name(choice):
    if choice is None:
        return "auto"
    if choice is True or choice == "any":
        return "required"
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = choice.get("function", {}).get("name")
        if isinstance(name, str) and name:
            return name
    raise GatewayError("unsupported_capability", "Unsupported tool_choice.")


class GatewayChatModel(BaseChatModel):
    gateway: Gateway = Field(exclude=True)
    profile: str | None = None

    @classmethod
    def from_config(cls, path="llm-gateway.yaml", *, profile=None, **kwargs):
        return cls(gateway=Gateway.from_config(path), profile=profile, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "local-llm-gateway"

    @property
    def _identifying_params(self):
        return {"profile": self.profile or self.gateway.config.default_profile}

    def bind_tools(self, tools: Sequence[Any], *, tool_choice=None, **kwargs):
        unknown = set(kwargs) - {"ls_structured_output_format", "structured_output_format"}
        if unknown:
            raise GatewayError(
                "unsupported_capability",
                f"Unsupported bind_tools options: {', '.join(sorted(unknown))}",
            )
        return self.bind(
            tools=[convert_to_openai_tool(tool) for tool in tools],
            tool_choice=tool_choice_name(tool_choice),
            **kwargs,
        )

    def _request(self, messages, stop, kwargs):
        unknown = set(kwargs) - {
            "tools",
            "tool_choice",
            "output_schema",
            "ls_structured_output_format",
            "structured_output_format",
        }
        if stop or unknown:
            raise GatewayError(
                "unsupported_capability",
                "Stop sequences and unrecognized model options are unsupported: "
                + ", ".join(sorted(unknown)),
            )
        converted = []
        roles = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
        for message in messages:
            role = roles.get(message.type)
            if role is None:
                raise GatewayError(
                    "unsupported_capability", f"Unsupported message type: {message.type}"
                )
            if isinstance(message, AIMessage) and message.invalid_tool_calls:
                raise GatewayError("invalid_request", "Conversation contains invalid tool calls.")
            converted.append(
                Message(
                    role=role,
                    content=text_content(message.content),
                    name=message.name,
                    tool_call_id=getattr(message, "tool_call_id", None),
                    tool_calls=[
                        ToolCall(id=c["id"], name=c["name"], arguments=c["args"])
                        for c in getattr(message, "tool_calls", [])
                    ],
                )
            )
        tools = []
        for raw in kwargs.get("tools", []):
            normalized = convert_to_openai_tool(raw)
            if normalized.get("type") != "function":
                raise GatewayError("unsupported_capability", "Only function tools are supported.")
            function = normalized["function"]
            tools.append(
                Tool(
                    name=function["name"],
                    description=function.get("description", ""),
                    parameters=function.get("parameters", {"type": "object"}),
                )
            )
        return GenerationRequest(
            messages=converted,
            tools=tools,
            tool_choice=tool_choice_name(kwargs.get("tool_choice")),
            output_schema=kwargs.get("output_schema"),
        )

    def _result(self, response):
        message = AIMessage(
            content=response.text,
            tool_calls=[
                {"name": c.name, "args": c.arguments, "id": c.id, "type": "tool_call"}
                for c in response.tool_calls
            ],
            response_metadata={
                "provider": response.provider,
                "requested_model": response.model,
                "reported_model": response.reported_model,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            },
            id=response.id,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        request = self._request(messages, stop, kwargs)
        return self._result(self.gateway.generate_sync(request, profile=self.profile))

    async def _agenerate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        request = self._request(messages, stop, kwargs)
        return self._result(await self.gateway.generate(request, profile=self.profile))
