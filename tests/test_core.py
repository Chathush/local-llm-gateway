import asyncio

import pytest
from pydantic import ValidationError

from local_llm_gateway import Gateway, GenerationRequest, Message, Tool
from local_llm_gateway.config import GatewayConfig
from local_llm_gateway.errors import GatewayError
from local_llm_gateway.providers.base import ProviderResult


def config(**runtime):
    return GatewayConfig.model_validate(
        {
            "default_profile": "main",
            "profiles": {"main": {"provider": "codex", "model": "test-model"}},
            "runtime": runtime,
        }
    )


class FixtureProvider:
    def __init__(self, answer, delay=0):
        self.answer = answer
        self.delay = delay
        self.active = 0
        self.peak = 0

    async def generate(self, request, model, runtime):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            return ProviderResult(payload=self.answer, reported_model=model)
        finally:
            self.active -= 1


def test_rejects_unknown_config_and_missing_default():
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate({"profiles": {}, "typo": True})
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(
            {"default_profile": "absent", "profiles": {"a": {"provider": "codex"}}}
        )


def test_rejects_orphan_tool_results_and_duplicate_tools():
    with pytest.raises(ValidationError):
        GenerationRequest(messages=[Message(role="tool", content="x", tool_call_id="missing")])
    with pytest.raises(ValidationError):
        GenerationRequest(
            messages=[Message(role="user", content="hi")], tools=[Tool(name="x"), Tool(name="x")]
        )


async def test_routes_and_validates_text():
    gateway = Gateway(
        config(), providers={"codex": FixtureProvider({"text": "hello", "tool_calls": []})}
    )
    response = await gateway.generate(
        GenerationRequest(messages=[Message(role="user", content="hi")])
    )
    assert response.text == "hello"
    assert response.provider == "codex"
    assert response.model == "test-model"
    assert response.finish_reason == "stop"


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "", "tool_calls": [{"name": "unknown", "arguments": "{}"}]},
        {"text": "", "tool_calls": [{"name": "add", "arguments": '{"a":"wrong"}'}]},
        {"text": "", "tool_calls": [{"name": "add", "arguments": "not json"}]},
    ],
)
async def test_rejects_bad_tool_calls(payload):
    gateway = Gateway(config(), providers={"codex": FixtureProvider(payload)})
    req = GenerationRequest(
        messages=[Message(role="user", content="calculate")],
        tools=[
            Tool(
                name="add",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                },
            )
        ],
    )
    with pytest.raises(GatewayError) as err:
        await gateway.generate(req)
    assert err.value.code == "invalid_provider_response"


async def test_tool_call_has_id_and_can_be_replayed():
    provider = FixtureProvider(
        {"text": "", "tool_calls": [{"name": "add", "arguments": '{"a":2}'}]}
    )
    gateway = Gateway(config(), providers={"codex": provider})
    req = GenerationRequest(
        messages=[Message(role="user", content="add")], tools=[Tool(name="add")]
    )
    res = await gateway.generate(req)
    assert res.tool_calls[0].arguments == {"a": 2}
    assert res.tool_calls[0].id
    replay = GenerationRequest(
        messages=[
            *req.messages,
            res.as_message(),
            Message(role="tool", content="3", tool_call_id=res.tool_calls[0].id),
        ]
    )
    assert replay.messages[-1].tool_call_id == res.tool_calls[0].id


async def test_enforces_schema_and_required_tool_choice():
    provider = FixtureProvider({"text": '{"answer":"wrong"}', "tool_calls": []})
    gateway = Gateway(config(), providers={"codex": provider})
    with pytest.raises(GatewayError):
        await gateway.generate(
            GenerationRequest(
                messages=[Message(role="user", content="hi")],
                output_schema={"type": "object", "properties": {"answer": {"type": "integer"}}},
            )
        )
    with pytest.raises(GatewayError):
        await gateway.generate(
            GenerationRequest(
                messages=[Message(role="user", content="hi")],
                tools=[Tool(name="add")],
                tool_choice="required",
            )
        )


async def test_concurrency_and_queue_bound():
    provider = FixtureProvider({"text": "ok", "tool_calls": []}, delay=0.05)
    gateway = Gateway(
        config(max_concurrency_per_provider=1, max_queue_per_provider=1),
        providers={"codex": provider},
    )
    req = GenerationRequest(messages=[Message(role="user", content="hi")])
    tasks = [asyncio.create_task(gateway.generate(req)) for _ in range(3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert provider.peak == 1
    assert sum(isinstance(r, GatewayError) and r.code == "queue_full" for r in results) == 1


async def test_timeout_releases_capacity():
    provider = FixtureProvider({"text": "ok", "tool_calls": []}, delay=0.1)
    gateway = Gateway(config(timeout_seconds=0.02), providers={"codex": provider})
    req = GenerationRequest(messages=[Message(role="user", content="hi")])
    with pytest.raises(GatewayError) as err:
        await gateway.generate(req)
    assert err.value.code == "timeout"
    assert provider.active == 0
    provider.delay = 0
    assert (await gateway.generate(req)).text == "ok"


def test_sync_api():
    provider = FixtureProvider({"text": "ok", "tool_calls": []})
    gateway = Gateway(config(), providers={"codex": provider})
    assert (
        gateway.generate_sync(GenerationRequest(messages=[Message(role="user", content="hi")])).text
        == "ok"
    )
