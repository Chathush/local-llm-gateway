# Local LLM Gateway

Use your installed **Codex** or **Claude Code** as the model connection for personal local applications. Select the provider/model in YAML. No provider API key is required; the installed CLI manages its subscription login.

```text
LangGraph / ADK / any Python app / HTTP client
                       |
              Local LLM Gateway
              config + validation
                 /           \
           Codex CLI       Claude Code CLI
                 |           |
              OpenAI       Anthropic
```

Models run remotely. Your framework owns its agent loop, tools, approvals, memory, and history. The gateway returns text or validated tool-call requests; it never executes application tools.

## Quick start

Requires Python **3.11+** and at least one installed, signed-in provider CLI. Tested on macOS; Linux process-group handling is implemented, Windows has not been verified.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[http,langgraph,adk]'
llm-gateway doctor
llm-gateway smoke --profile primary
llm-gateway smoke --profile primary --tool
```

This repository includes `llm-gateway.yaml`. In another application, run `llm-gateway init .` to create config and a portable integration guide. Existing files are preserved.

Sign in, when needed, using `codex login` or `claude auth login`. `doctor` checks executable support and subscription login without calling a model. `smoke` consumes subscription usage; `--tool` makes two calls and performs a local addition tool round trip. `models` lists configured aliases, not account entitlements.

## Configure each application

```yaml
schema_version: 1
application: my-local-agent
default_profile: primary
profiles:
  primary:
    provider: codex
    model: null
  claude:
    provider: claude_code
    model: sonnet
```

Use a model available through your CLI and subscription. `null` selects the runtime's built-in default; the gateway suppresses personal/project runtime configuration except authentication and administrator policies. Aliases can change; use an explicit model ID for reproducibility. The checked-in config selects the Codex model verified on this machine; change it for another account if needed.

See [the full config template](src/local_llm_gateway/resources/example.yaml) for timeouts, concurrency/queue bounds, executable paths, and input/output limits. Unknown settings fail validation. Provider configuration is never rewritten.

## LangChain and LangGraph

```python
from local_llm_gateway.integrations.langchain import GatewayChatModel

model = GatewayChatModel.from_config("llm-gateway.yaml", profile="primary")
response = await model.bind_tools(my_tools).ainvoke(messages)
```

Use it in a LangGraph model node with a `ToolNode`, or a compatible LangChain agent. Supports `invoke`, `ainvoke`, `bind_tools`, and tool-based `with_structured_output`. [Complete LangGraph example](examples/langgraph_agent.py).

## Google ADK

```python
from google.adk.agents import LlmAgent
from local_llm_gateway.integrations.adk import GatewayLlm

agent = LlmAgent(
    name="assistant",
    model=GatewayLlm.from_config("llm-gateway.yaml", profile="claude"),
    instruction="Help the user using the available tools.",
    tools=my_tools,
)
```

Use non-streaming execution. Tested with ADK 2.11's real runner. [ADK example](examples/adk_agent.py).

## Any Python framework

```python
from local_llm_gateway import Gateway, GenerationRequest, Message

gateway = Gateway.from_config("llm-gateway.yaml")
response = await gateway.generate(
    GenerationRequest(messages=[Message(role="user", content="Explain dependency injection.")])
)
print(response.text)
```

Reuse one gateway per application. `generate_sync` works outside running event loops. For another framework, translate its messages and tool definitions into this contract and translate the response back. [Core example](examples/basic_chat.py) · [Integration guide](INTEGRATION.md).

## Other languages / OpenAI-compatible clients

```sh
llm-gateway serve --port 8787
```

The server creates `.gateway-token` with restrictive permissions. Set the client's base URL to `http://127.0.0.1:8787/v1`, API key to that file's contents, model to `primary` or `claude`, and streaming off. This is a local access token, not a provider key.

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",
    api_key=Path(".gateway-token").read_text().strip(),
)
result = client.chat.completions.create(
    model="primary",
    messages=[{"role": "user", "content": "Hello"}],
)
print(result.choices[0].message.content)
```

Install `openai` to run this snippet. A [dependency-free Node client](examples/http_client.mjs) uses `fetch`.

| Route | Contract |
| --- | --- |
| `GET /health` | Process health only; does not call providers |
| `GET /v1/models` | Configured profile aliases; authenticated |
| `POST /v1/chat/completions` | Text, function tools/results, tool choice, JSON output; `stream=false`, `n=1` |
| `POST /generate` | Native GenerationRequest fields plus optional `profile` |

Chat Completions accepts `response_format` types `text`, `json_object`, and `json_schema`. `tool_choice` accepts `auto`, `none`, `required`, or a forced function. Unknown fields fail. `usage` is null when unreported. Select your framework's Chat Completions backend, not its Responses API backend.

## Take it to another codebase

```sh
pip install '/absolute/path/to/this/repository[langgraph]'
cd /path/to/your/application
llm-gateway init .
```

Then tell Codex or Claude Code:

> Read LLM_GATEWAY_INTEGRATION.md and integrate this application with local-llm-gateway. Preserve the framework and tool loop. Put provider/model choices in llm-gateway.yaml. Verify a text call and a tool round trip.

Add a guide reference to the application's existing `AGENTS.md` or `CLAUDE.md`. For reproducible distribution, build/install a wheel:

```sh
python -m pip install build
python -m build
pip install '/path/to/local_llm_gateway-0.1.0-py3-none-any.whl[adk]'
```

## Capabilities and limits

| Supported in v0.1 | Explicitly unsupported |
| --- | --- |
| Text, JSON Schema output, multi-turn history | Images, audio, embeddings, realtime/live sessions |
| Validated application function calls | Provider-hosted tools, coding-agent task delegation |
| Named model profiles | Automatic failover, account rotation, API-key fallback |
| Async calls, bounded queues, cancellation | Token streaming, sampling controls, raw API parity |

Tool calling is **structured-output emulation**. The CLI generates an envelope of tool decisions; the gateway validates tool names and JSON arguments before your framework receives them. CLI behavior differs from raw model APIs. Evaluate quality and latency for your application.

Each call starts a fresh runtime session and replays supplied history. This trades speed/context efficiency for predictable state ownership. Return every requested tool result with its matching ID. Oversized input is rejected, not silently truncated. Provider context limits can be smaller than the gateway's byte limit.

LangChain's inherited stream methods can yield one completed message; there is no token streaming. HTTP and ADK incremental streaming requests fail explicitly.

## Authentication, isolation, and troubleshooting

- CLIs authenticate normally; the gateway never reads/copies token files.
- API keys, alternate endpoints, or alternate-provider environment settings cause `auth_conflict`. Unset them for the gateway process.
- Authentication is checked before each request. Expired sessions produce `authentication_required`; sign in through the CLI yourself.
- Codex runs read-only in a temporary directory with shell, browser, app, multi-agent, plugin, hook, and host-skill controls. Claude uses `--safe-mode`, no built-in tools/MCP servers, and denied permission prompts. Required controls are checked before inference.
- This is not an OS security boundary. Administrator policy can affect runtimes; use only trusted local applications. Runtime tools appearing in Codex output are treated as an error.
- Prompts and raw provider outputs are not logged by the gateway. Provider logs/retention are separate. Sessions are requested as ephemeral/nonpersistent.
- Timeout includes queue time. Concurrency applies per provider per Gateway instance; different processes have different queues.
- Subscription limits are shared with interactive use. Account settings determine extra-usage billing; the gateway does not configure those settings.

Stable error codes include `authentication_required`, `auth_conflict`, `unsupported_runtime`, `model_unavailable`, `rate_limited`, `queue_full`, `timeout`, `input_too_large`, `output_too_large`, `invalid_provider_response`, and `unsupported_capability`.

## Development

```sh
python -m pip install -e '.[dev,http,langgraph,adk]'
pytest
ruff check .
ruff format --check .
python -m build
```

Default tests use deterministic provider fixtures, real subprocesses, and real framework runners; no subscription usage. Live checks are explicit: `llm-gateway smoke` or `GATEWAY_LIVE_PROFILE=primary pytest tests/test_live.py`.

[Design](docs/design.md) · [Plan](docs/implementation-plan.md) · [Verification](docs/verification.md).

Provider references: [Codex automation](https://learn.chatgpt.com/docs/non-interactive-mode), [Codex authentication](https://learn.chatgpt.com/docs/auth), [Claude programmatic execution](https://code.claude.com/docs/en/headless), [Claude credential-use rules](https://code.claude.com/docs/en/legal-and-compliance).

For personal local experiments. This package does not offer subscription access to other users or replace providers' commercial APIs.
