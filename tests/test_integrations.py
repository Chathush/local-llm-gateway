import pytest
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from local_llm_gateway import Gateway
from local_llm_gateway.config import GatewayConfig
from local_llm_gateway.errors import GatewayError
from local_llm_gateway.integrations.adk import GatewayLlm
from local_llm_gateway.integrations.langchain import GatewayChatModel
from local_llm_gateway.providers.base import ProviderResult


class ToolRoundTrip:
    def __init__(self):
        self.requests = []

    async def generate(self, request, model, runtime):
        self.requests.append(request)
        if any(m.role == "tool" for m in request.messages):
            result = next(m.content for m in reversed(request.messages) if m.role == "tool")
            assert "5" in result
            return ProviderResult(payload={"text": "The answer is 5.", "tool_calls": []})
        assert any(t.name == "add" for t in request.tools)
        return ProviderResult(
            payload={"text": "", "tool_calls": [{"name": "add", "arguments": '{"a":2,"b":3}'}]}
        )


def gateway(provider):
    return Gateway(
        GatewayConfig.model_validate({"profiles": {"primary": {"provider": "codex"}}}),
        providers={"codex": provider},
    )


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def test_langgraph_owns_tool_execution_and_replays_result():
    provider = ToolRoundTrip()
    model = GatewayChatModel(gateway=gateway(provider)).bind_tools([add])

    async def call_model(state):
        return {"messages": [await model.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([add]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    result = await graph.compile().ainvoke({"messages": [HumanMessage(content="What is 2 + 3?")]})
    assert result["messages"][-1].content == "The answer is 5."
    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_result.tool_call_id == result["messages"][1].tool_calls[0]["id"]
    assert len(provider.requests) == 2


def test_langchain_sync_bind_tools_and_options():
    model = GatewayChatModel(gateway=gateway(ToolRoundTrip()))
    result = model.bind_tools([add], tool_choice="add").invoke("Calculate")
    assert result.tool_calls[0]["args"] == {"a": 2, "b": 3}
    with pytest.raises(GatewayError):
        model.invoke("hi", temperature=0.2)
    with pytest.raises(GatewayError):
        model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}}
                    ]
                )
            ]
        )


def test_langchain_structured_output_parser():
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    class StructuredProvider:
        async def generate(self, request, model, runtime):
            assert request.tool_choice == "required"
            assert request.tools[0].name == "Answer"
            return ProviderResult(
                payload={"text": "", "tool_calls": [{"name": "Answer", "arguments": '{"value":5}'}]}
            )

    model = GatewayChatModel(gateway=gateway(StructuredProvider()))
    assert model.with_structured_output(Answer).invoke("Calculate").value == 5


async def test_adk_runner_owns_tool_execution_and_replays_result():
    provider = ToolRoundTrip()

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    agent = LlmAgent(
        name="calculator",
        model=GatewayLlm(gateway=gateway(provider)),
        instruction="Calculate using add.",
        tools=[add],
    )
    runner = InMemoryRunner(agent=agent, app_name="gateway_test")
    session = await runner.session_service.create_session(app_name="gateway_test", user_id="test")
    events = [
        event
        async for event in runner.run_async(
            user_id="test",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="2 + 3?")]),
        )
    ]
    assert any(
        event.is_final_response() and event.content.parts[0].text == "The answer is 5."
        for event in events
    )
    assert len(provider.requests) == 2
    assert provider.requests[0].messages[0].role == "system"


async def test_adk_rejects_sampling_options():
    from google.adk.models import LlmRequest

    model = GatewayLlm(gateway=gateway(ToolRoundTrip()))
    request = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="hello")])],
        config=types.GenerateContentConfig(temperature=0.5),
    )
    with pytest.raises(GatewayError):
        async for _ in model.generate_content_async(request):
            pass


def test_adk_schema_conversion_preserves_property_names_and_literal_data():
    from local_llm_gateway.integrations.adk import json_schema

    schema = types.Schema(
        type="OBJECT",
        properties={
            "nullable": types.Schema(type="BOOLEAN"),
            "propertyOrdering": types.Schema(type="STRING"),
        },
        required=["nullable"],
    )
    assert json_schema(schema) == {
        "type": "object",
        "properties": {"nullable": {"type": "boolean"}, "propertyOrdering": {"type": "string"}},
        "required": ["nullable"],
    }
    assert json_schema(
        {
            "type": "OBJECT",
            "default": {"type": "UPPER", "nullable": True},
            "properties": {"type": {"type": "STRING", "enum": ["OBJECT"]}},
        }
    ) == {
        "type": "object",
        "default": {"type": "UPPER", "nullable": True},
        "properties": {"type": {"type": "string", "enum": ["OBJECT"]}},
    }
