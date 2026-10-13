import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from local_llm_gateway.config import ProviderConfig, RuntimeConfig
from local_llm_gateway.contracts import GenerationRequest, Message
from local_llm_gateway.errors import GatewayError
from local_llm_gateway.providers.cli import CLIProvider, parse_claude, parse_codex
from local_llm_gateway.providers.process import run_process


async def test_process_stdin_preserves_shell_characters(tmp_path):
    prompt = "hello $(touch BAD) `echo x` ' \n world"
    result = await run_process(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
        stdin=prompt,
        cwd=tmp_path,
        env=os.environ.copy(),
        limit=1024,
    )
    assert result.stdout.strip() == prompt
    assert not (tmp_path / "BAD").exists()


async def test_oversized_process_is_stopped(tmp_path):
    with pytest.raises(GatewayError) as err:
        await run_process(
            [sys.executable, "-c", "print('x'*5000)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            limit=1000,
        )
    assert err.value.code == "output_too_large"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group assertion")
async def test_cancellation_terminates_process(tmp_path):
    pid_file = tmp_path / "pid"
    code = (
        "import os,time,pathlib; pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(60)"
    )
    task = asyncio.create_task(
        run_process([sys.executable, "-c", code], cwd=tmp_path, env=os.environ.copy(), limit=1024)
    )
    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(0.01)
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_codex_parser_requires_completed_turn():
    payload = {"text": "ok", "tool_calls": []}
    messages = [
        {"type": "thread.started", "thread_id": "one"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(payload)}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}},
    ]
    result = parse_codex("\n".join(map(json.dumps, messages)))
    assert result.payload == payload
    assert result.usage["input_tokens"] == 10
    with pytest.raises(GatewayError):
        parse_codex("\n".join(map(json.dumps, messages[:-1])))


def test_provider_errors_classify_limits_and_models_without_echoing_output():
    from local_llm_gateway.providers.cli import provider_error

    assert provider_error("rate_limit_error private prompt").code == "rate_limited"
    assert provider_error("model is not available for this account").code == "model_unavailable"
    assert "private prompt" not in str(provider_error("error private prompt"))


def test_codex_does_not_accept_runtime_tool_execution():
    with pytest.raises(GatewayError) as err:
        parse_codex(
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "command": "echo x"},
                }
            )
        )
    assert err.value.code == "unsupported_runtime"


def test_claude_structured_result_and_error():
    result = parse_claude(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": {"text": "yes", "tool_calls": []},
                "usage": {"input_tokens": 5},
                "modelUsage": {"claude-test": {}},
            }
        )
    )
    assert result.payload["text"] == "yes"
    assert result.reported_model == "claude-test"
    with pytest.raises(GatewayError):
        parse_claude(
            json.dumps(
                {
                    "type": "result",
                    "subtype": "error_max_turns",
                    "is_error": True,
                    "result": "partial",
                }
            )
        )


@pytest.mark.parametrize("bad", ["not json", "[]", "null", '{"type":"result"}'])
def test_malformed_claude_is_not_success(bad):
    with pytest.raises(GatewayError):
        parse_claude(bad)


def fake_cli(tmp_path, provider):
    executable = tmp_path / provider
    executable.write_text(f"""#!{Path(sys.executable).resolve()}
import json,sys
args=sys.argv[1:]
if "--help" in args:
    print("--json --output-schema --ephemeral --ignore-user-config --ignore-rules --safe-mode --tools --strict-mcp-config --no-session-persistence --json-schema --permission-mode")
elif "features" in args:
    print("shell_tool stable true\\napps stable true\\nhooks stable true\\nplugins stable true\\nmulti_agent stable true\\nskip_host_skill_discovery experimental false")
elif "--version" in args:
    print("test-version")
elif "status" in args:
    print("Logged in using ChatGPT" if "login" in args else json.dumps({{"loggedIn":True,"authMethod":"claude.ai","apiProvider":"firstParty"}}))
else:
    incoming = json.loads(sys.stdin.read().split("APPLICATION_REQUEST\\n")[-1])
    envelope = {{"text":incoming["messages"][-1]["content"],"tool_calls":[]}}
    if {provider!r} == "codex":
        print(json.dumps({{"type":"item.completed","item":{{"type":"agent_message","text":json.dumps(envelope)}}}}))
        print(json.dumps({{"type":"turn.completed","usage":{{"input_tokens":1}}}}))
    else:
        print(json.dumps({{"type":"result","subtype":"success","is_error":False,"structured_output":envelope}}))
""")
    executable.chmod(0o755)
    return executable


@pytest.mark.parametrize("name", ["codex", "claude_code"])
async def test_provider_round_trip_real_subprocess(tmp_path, monkeypatch, name):
    for key in list(os.environ):
        if key.startswith(("OPENAI_", "CODEX_", "ANTHROPIC_", "CLAUDE_")):
            monkeypatch.delenv(key, raising=False)
    provider = CLIProvider(name, ProviderConfig(executable=str(fake_cli(tmp_path, name))))
    result = await provider.generate(
        GenerationRequest(messages=[Message(role="user", content="roundtrip ✓")]),
        "test-model",
        RuntimeConfig(),
    )
    assert result.payload["text"] == "roundtrip ✓"


async def test_conflicting_api_auth_fails_before_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-secret")
    provider = CLIProvider(
        "claude_code", ProviderConfig(executable=str(fake_cli(tmp_path, "claude_code")))
    )
    with pytest.raises(GatewayError) as err:
        await provider.generate(
            GenerationRequest(messages=[Message(role="user", content="hi")]), None, RuntimeConfig()
        )
    assert err.value.code == "auth_conflict"
    assert "not-a-real-secret" not in str(err.value)
