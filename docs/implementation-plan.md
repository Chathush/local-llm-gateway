# Local LLM Gateway Implementation Plan

> For agentic workers: use superpowers:executing-plans to implement in this session.

**Goal:** Build the approved reusable subscription-backed local gateway in the requested folder.

**Architecture:** Typed core and subprocess providers with optional framework adapters and HTTP transport. Application-owned state and tools; structured-output tool decisions.

**Tech Stack:** Python 3.11+, Pydantic, JSON Schema, PyYAML; optional FastAPI, LangChain Core, LangGraph, Google ADK.

**Spec:** `docs/design.md`.

## Global constraints

- Personal local use; preserve installed CLI authentication; no token extraction.
- Reject unsupported capabilities; never silently fall back to API billing.
- No framework dependencies in core; no provider tools executing application actions.
- Use the folder requested by the user (new, empty, no existing Git repository).

## Review focus

- Tool results retain matching IDs across framework and HTTP boundaries.
- Queued, timed-out, cancelled, or oversized subprocesses release capacity and terminate.
- Unknown provider output and auth changes cannot be reported as success.
- Tool schemas, constrained tool choice, and structured final responses are enforced.
- HTTP authentication, body bounds, unsupported options, and local-only startup hold.

## Tasks

1. [x] Core: contracts.py, config.py, errors.py, protocol.py, gateway.py; test input validation, schemas, profile selection, response validation, concurrency. Interface: GenerationRequest -> GenerationResponse.
2. [x] Providers: providers/process.py, providers/cli.py; test actual fake executable processes, output parsing, nonzero exits, auth, timeout/cancellation, output size. Interface: Provider.generate(request, model, runtime).
3. [x] Integrations: integrations/langchain.py, integrations/adk.py; test with real LangGraph and ADK runners and deterministic external provider fixtures. Interface: GatewayChatModel and GatewayLlm wrap Gateway.
4. [x] HTTP/CLI: server.py, cli.py; test authenticated requests, OpenAI tool conversion, error mapping, CLI init and doctor. Interface: installed llm-gateway command and app factory.
5. [x] Delivery: README, INTEGRATION, examples, example config, packaged resources; build wheel, run full suite/lint, real CLI smoke calls, separate installation test, independent final code review.

Each task uses failing tests first, implementation, then the full suite. No publication, remote push, or account configuration changes.
