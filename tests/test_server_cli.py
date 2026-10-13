import json
import os
import subprocess
import sys

import httpx
import pytest

from local_llm_gateway import Gateway
from local_llm_gateway.config import GatewayConfig
from local_llm_gateway.providers.base import ProviderResult
from local_llm_gateway.server import create_app


class TextProvider:
    async def generate(self, request, model, runtime):
        if request.tools and request.tool_choice != "none":
            return ProviderResult(
                payload={
                    "text": "",
                    "tool_calls": [{"name": "lookup", "arguments": '{"id":"123"}'}],
                }
            )
        return ProviderResult(payload={"text": "hello", "tool_calls": []})


@pytest.fixture
def app():
    gateway = Gateway(
        GatewayConfig.model_validate({"profiles": {"primary": {"provider": "codex"}}}),
        providers={"codex": TextProvider()},
    )
    return create_app(gateway, token="local-test-token")


async def test_auth_and_openai_text(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/v1/models")).status_code == 401
        client.headers["Authorization"] = "Bearer local-test-token"
        assert (await client.get("/v1/models")).json()["data"][0]["id"] == "primary"
        result = await client.post(
            "/v1/chat/completions",
            json={"model": "primary", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert result.status_code == 200
        assert result.json()["choices"][0]["message"]["content"] == "hello"
        assert result.json()["usage"] is None


async def test_official_openai_client_consumes_http_contract(app):
    from openai import AsyncOpenAI

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as transport:
        client = AsyncOpenAI(
            base_url="http://localhost/v1", api_key="local-test-token", http_client=transport
        )
        result = await client.chat.completions.create(
            model="primary", messages=[{"role": "user", "content": "hi"}]
        )
        assert result.choices[0].message.content == "hello"
        assert result.model == "primary"


async def test_openai_tool_call_and_replay(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer local-test-token"},
    ) as client:
        messages = [{"role": "user", "content": "lookup 123"}]
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {"type": "object"}},
                    }
                ],
            },
        )
        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
        call = message["tool_calls"][0]
        assert json.loads(call["function"]["arguments"]) == {"id": "123"}
        result = await client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [
                    *messages,
                    message,
                    {"role": "tool", "tool_call_id": call["id"], "content": "found"},
                ],
            },
        )
        assert result.status_code == 200


@pytest.mark.parametrize(
    "options",
    [
        {"stream": True},
        {"temperature": 0.5},
        {"n": 2},
        {"parallel_tool_calls": False},
        {"logprobs": True},
    ],
)
async def test_unsupported_options_fail_instead_of_ignored(app, options):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer local-test-token"},
    ) as client:
        result = await client.post(
            "/v1/chat/completions",
            json={"model": "primary", "messages": [{"role": "user", "content": "hi"}], **options},
        )
        assert result.status_code == 400
        assert result.json()["error"]["code"] in {"unsupported_capability", "invalid_request"}


async def test_http_errors_do_not_echo_prompts(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer local-test-token"},
    ) as client:
        result = await client.post(
            "/v1/chat/completions",
            json={
                "model": "primary",
                "messages": [{"role": "secret-prompt", "content": "secret-body"}],
            },
        )
        assert result.status_code == 400
        assert "secret" not in result.text
        result = await client.post("/generate", content=b"invalid-json")
        assert result.status_code == 400
        large = await client.post("/generate", content=b"x" * 1_100_000)
        assert large.status_code == 413


def test_init_does_not_overwrite_existing_files(tmp_path):
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-m", "local_llm_gateway", "init", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    config = tmp_path / "llm-gateway.yaml"
    assert GatewayConfig.from_file(config).default_profile == "primary"
    assert (tmp_path / "LLM_GATEWAY_INTEGRATION.md").exists()
    config.write_text("keep-me")
    again = subprocess.run(
        [sys.executable, "-m", "local_llm_gateway", "init", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert config.read_text() == "keep-me"
    assert again.returncode == 0
