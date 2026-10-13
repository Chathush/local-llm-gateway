"""Explicit opt-in only: GATEWAY_LIVE_PROFILE=primary pytest tests/test_live.py."""

import os

import pytest

from local_llm_gateway import Gateway, GenerationRequest, Message, Tool

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("GATEWAY_LIVE_PROFILE"), reason="Live inference is opt-in"),
]


async def test_live_json_and_tool_round_trip():
    gateway = Gateway.from_config()
    profile = os.environ["GATEWAY_LIVE_PROFILE"]
    request = GenerationRequest(
        messages=[
            Message(
                role="user",
                content="Use add with a=2 and b=3. After the result, return JSON with answer equal to the numeric result.",
            )
        ],
        tools=[
            Tool(
                name="add",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            )
        ],
        tool_choice="add",
    )
    first = await gateway.generate(request, profile=profile)
    assert first.tool_calls
    history = [*request.messages, first.as_message()]
    for call in first.tool_calls:
        history.append(
            Message(
                role="tool",
                content=str(call.arguments["a"] + call.arguments["b"]),
                tool_call_id=call.id,
            )
        )
    final = await gateway.generate(
        GenerationRequest(
            messages=history,
            tool_choice="none",
            output_schema={
                "type": "object",
                "properties": {"answer": {"const": 5}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        ),
        profile=profile,
    )
    assert final.json_output() == {"answer": 5}
