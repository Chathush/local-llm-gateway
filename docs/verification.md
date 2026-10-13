# Verification record

Verified locally on macOS, 2026-10-13, with Python 3.12.

## Automated verification

- Full default suite: **45 passed, 1 skipped**. Live inference is skipped unless explicitly enabled.
- Real LangGraph and Google ADK runners completed application-owned tool round trips with deterministic provider fixtures.
- Claude and Codex adapters passed subprocess contract tests using executable fixtures.
- The official OpenAI AsyncOpenAI client consumed the HTTP interface successfully.
- Cancellation, queue limits, output bounds, tool IDs, schemas, authentication conflicts, required CLI controls, and unsupported HTTP options are covered.
- Ruff lint/format checks passed; `pip check` reported no broken requirements.
- ADK emits one upstream warning about experimental JSON_SCHEMA_FOR_FUNC_DECL; the test passes.

## Live providers

- Codex CLI `0.154.0-alpha.6.2`, ChatGPT authentication, model `gpt-6-astra`: a text request returned `gateway-ok`.
- The opt-in live test passed a complete tool call/result round trip and final JSON Schema validation (`{"answer": 5}`).
- Later live runs of the complete LangGraph and ADK examples returned the provider's usage-limit error. Their framework paths pass automated tests, but these live example runs did not complete. No automatic retries or provider changes were made.
- Claude Code `2.1.268`: initial status reported subscription login, but live execution could not complete. Subsequent native status reports an expired login; the gateway reports `authentication_required`. Claude live inference remains unverified until `claude auth login` is completed.
- Doctor establishes CLI/auth readiness, not model entitlement or remaining quota.

## Distribution

- Built `dist/local_llm_gateway-0.1.0-py3-none-any.whl` and source archive.
- Installed the wheel with HTTP extras in a fresh virtual environment.
- In another temporary application folder, ran init, loaded config, listed model profiles, and created the HTTP app.
- Confirmed imports came from installed site-packages, not this source checkout.
- Config and integration documentation are bundled in the wheel.

## Independent review

A separate read-only reviewer identified an ADK schema traversal bug involving property names such as `nullable`. A failing regression test reproduced it; schema-aware traversal fixed it and the full suite passed afterward. No other concrete blockers were reported. Provider internals, OS isolation guarantees, and all dependency-version combinations were outside that review's evidence.

## Tested dependencies

- google-adk 2.11.0 (declared minimum now 2.11)
- langchain-core 1.6.10
- langgraph 1.2.15
- FastAPI 0.141.1
- Pydantic 2.14.1

`requirements-dev.lock` records the local verification environment. It is a developer snapshot; consumers should pin their own dependencies. Windows process-tree behavior and older runtimes were not verified. Unsupported CLI controls fail preflight.
