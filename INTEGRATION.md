# Integrating Local LLM Gateway

This component uses the machine's installed Codex or Claude Code and their own subscription login. Models run remotely. The gateway runs locally. Use for personal local experiments.

## Install and configure

Python 3.11+ is required. Install a pinned gateway wheel or this repository with the desired extras:

```sh
pip install '/absolute/path/to/local-llm-gateway[langgraph,adk,http]'
llm-gateway init .
llm-gateway doctor
llm-gateway smoke --profile primary --tool
```

Set provider and model under `profiles` in `llm-gateway.yaml`. `model: null` uses the provider's default, not the user's project config. Only configured aliases are exposed over HTTP. There is no automatic fallback or provider API-key support. Sign in using `codex login` or `claude auth login` if needed. Do not copy credentials into this application.

## LangChain / LangGraph

```python
from local_llm_gateway.integrations.langchain import GatewayChatModel

model = GatewayChatModel.from_config("llm-gateway.yaml", profile="primary")
bound = model.bind_tools(application_tools)
response = await bound.ainvoke(messages)
```

Use this model in an existing LangGraph model node; use the framework's ToolNode/agent loop to execute tools. `invoke` is available for synchronous callers. `with_structured_output` uses LangChain's function-calling parser. Native token streaming is unavailable; LangChain's default stream returns a single completed message.

## Google ADK

```python
from google.adk.agents import LlmAgent
from local_llm_gateway.integrations.adk import GatewayLlm

agent = LlmAgent(
    name="assistant",
    model=GatewayLlm.from_config("llm-gateway.yaml", profile="primary"),
    instruction="Help the user.",
    tools=application_tools,
)
```

Use non-streaming ADK execution. Sampling controls and provider-hosted tools are unsupported; ordinary function tools and JSON output schemas are supported. ADK labels remain framework-side metadata.

## Any Python framework

```python
from local_llm_gateway import Gateway, GenerationRequest, Message

gateway = Gateway.from_config("llm-gateway.yaml")
response = await gateway.generate(
    GenerationRequest(messages=[Message(role="user", content="Explain this result.")])
)
print(response.text)
```

Translate your framework messages/tools into GenerationRequest, then translate GenerationResponse back. Tools use JSON Schema object parameters. Preserve tool-call IDs; append `response.as_message()` and one `Message(role='tool', tool_call_id=call.id, content=...)` per result. Gateway never executes application tools. Use `generate_sync` only outside an active asyncio loop.

## Other languages and OpenAI-compatible clients

```sh
llm-gateway serve --config llm-gateway.yaml --port 8787
```

Base URL: `http://127.0.0.1:8787/v1`. Use the contents of `.gateway-token` as the client's API key (a local access token, not a provider key), and `primary` or another configured profile as the model. Supports `/v1/models` and non-streaming `/v1/chat/completions` with text messages, function tools, tool results, `tool_choice`, and JSON response formats. `/generate` accepts the native contract plus optional `profile`.

No Responses API, embeddings, audio/images, sampling controls, token limits, parallel-tool-call control, provider-hosted tools, or incremental token streaming. Unknown/unsupported fields return errors. Function calling is validated structured-output emulation, not native raw-model API parity. Keep the service on loopback; do not expose it publicly.

## Instructions for coding agents

- Read this guide before integration. Change model setup, not the application workflow.
- Keep history, tools, memory, tool approvals, and loops in the application framework.
- Reuse one Gateway/model instance per application to share its bounded queue.
- Never add raw provider API calls or silently switch provider/model/authentication.
- Do not edit the gateway internals to integrate a new app; implement a thin adapter.
- Preserve tool IDs, all required tool results, and system instructions. Reject unsupported modalities/options explicitly.
- Verify a text call and a full framework-owned tool round trip. Tests use provider fixtures by default; live smoke calls consume subscription usage.
- Add a short reference to this file in existing AGENTS.md/CLAUDE.md; preserve existing instructions.

## Operational limits

Calls start fresh provider sessions and replay caller-supplied history. Requests share subscription limits with interactive use. Timeout covers queue and execution; queued requests are bounded per provider per Gateway instance, not across processes. Cancellation terminates subprocess groups on macOS/Linux. Windows process-tree behavior has not been verified. Prompts/responses are not logged by the gateway, but installed runtimes and remote providers have their own data handling. CLI guardrails are not an OS isolation boundary; use only trusted local applications.

Codex requires `--ignore-user-config`, `--ignore-rules`, `--ephemeral`, and JSON output support. Claude requires `--safe-mode` plus JSON schema and tool-control flags. The gateway checks required flags and subscription status before calls. Runtime upgrades can change behavior: run doctor and smoke after upgrades.
