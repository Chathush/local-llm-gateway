# Local LLM Gateway design

Approved conversational design, 2026-10-13. The user requested implementation in this folder and support for agentic frameworks including LangGraph and ADK.

## Scope

A Python 3.11+ package with a framework-neutral, text-only generation contract; installed Codex and Claude Code subprocess providers; optional LangChain/LangGraph and ADK adapters; and an authenticated loopback HTTP service with a documented OpenAI Chat Completions subset. No provider API key, credential extraction, automatic billing fallback, or provider-side application tool execution.

The application owns conversation history, tools, agent loops, and memory. Each provider call uses a fresh temporary working directory and an explicit JSON response envelope. Tool requests are structured-output emulation, validated locally, and returned to the application. They are never executed by the gateway. Provider built-in tools and customizations are disabled where supported; unsupported CLI versions fail preflight. The runtime is not an OS security boundary and remains suitable only for trusted local personal use.

## Interfaces

`Gateway.from_config(path)` loads named profiles. `await gateway.generate(GenerationRequest(...), profile=None)` and `gateway.generate_sync(...)` return `GenerationResponse`. Messages retain roles, tool call IDs, tool names, and results. Tools use JSON Schema. Output schema and tool choice are validated before and after inference. No token streaming or multimodal input in v0.1.

`GatewayChatModel` implements LangChain BaseChatModel (invoke, ainvoke, bind_tools) and works in LangGraph tool loops. `GatewayLlm` implements ADK BaseLlm. HTTP provides `/health`, `/v1/models`, `/v1/chat/completions`, and `/generate`; model names are configured profile aliases. Unsupported fields fail explicitly.

## Operations

Subscription login is verified through each CLI's status command without reading token files. Child environments omit conflicting API/provider settings; detected conflicts fail with actionable errors. Native CLI authentication remains untouched. Timeouts include queue time. Concurrency and queue lengths are bounded per provider per Gateway instance. Output is bounded, child processes cleaned up on cancellation, and secrets/prompts are not logged. HTTP uses a local bearer token and loopback binding.

## Acceptance

Text, JSON, and tool round trips work with provider fixtures; real LangGraph and ADK runners execute framework-owned tools. Real CLI smoke checks verify both providers when account and network state permit. Separate live checks are opt-in in the test suite. A wheel installs into another environment and provides packaged integration docs and example config.
