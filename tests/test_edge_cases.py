import asyncio
import os

import pytest
from pydantic import ValidationError

from local_llm_gateway import Gateway, GenerationRequest, Message, Tool
from local_llm_gateway.config import GatewayConfig, ProviderConfig, RuntimeConfig
from local_llm_gateway.errors import GatewayError
from local_llm_gateway.providers.base import ProviderResult
from local_llm_gateway.providers.cli import CLIProvider


def test_remote_schema_reference_is_rejected_before_inference():
    with pytest.raises(ValidationError):
        Tool(
            name="x",
            parameters={
                "type": "object",
                "properties": {"x": {"$ref": "https://example.invalid/schema"}},
            },
        )


async def test_cancelled_queued_request_releases_capacity():
    class SlowProvider:
        async def generate(self, request, model, runtime):
            await asyncio.sleep(0.05)
            return ProviderResult(payload={"text": "ok", "tool_calls": []})

    gateway = Gateway(
        GatewayConfig.model_validate(
            {
                "profiles": {"primary": {"provider": "codex"}},
                "runtime": {"max_queue_per_provider": 1},
            }
        ),
        providers={"codex": SlowProvider()},
    )
    req = GenerationRequest(messages=[Message(role="user", content="hello")])
    first = asyncio.create_task(gateway.generate(req))
    await asyncio.sleep(0.005)
    queued = asyncio.create_task(gateway.generate(req))
    await asyncio.sleep(0.005)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert (await gateway.generate(req)).text == "ok"
    await first


async def test_sync_api_reports_async_misuse():
    gateway = Gateway(
        GatewayConfig.model_validate({"profiles": {"primary": {"provider": "codex"}}}), providers={}
    )
    with pytest.raises(GatewayError) as error:
        gateway.generate_sync(GenerationRequest(messages=[Message(role="user", content="hi")]))
    assert error.value.code == "async_context"


async def test_codex_refuses_missing_isolation_feature(tmp_path, monkeypatch):
    from test_providers import fake_cli

    executable = fake_cli(tmp_path, "codex")
    for key in list(os.environ):
        if key.startswith(("OPENAI_", "CODEX_")):
            monkeypatch.delenv(key, raising=False)
    text = executable.read_text()
    text = text.replace("skip_host_skill_discovery", "missing_isolation")
    executable.write_text(text)
    provider = CLIProvider("codex", ProviderConfig(executable=str(executable)))
    with pytest.raises(GatewayError) as error:
        await provider.generate(
            GenerationRequest(messages=[Message(role="user", content="hi")]),
            "test",
            RuntimeConfig(),
        )
    assert error.value.code == "unsupported_runtime"
