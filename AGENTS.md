# Repository instructions

This is a Python local LLM gateway. Read README.md and INTEGRATION.md before changes.

- Keep core independent of agent frameworks; optional adapters belong in integrations/.
- Application frameworks own history, tools, approvals, and loops.
- Never extract CLI tokens, add automatic API billing fallback, or silently ignore unsupported capabilities.
- Subprocesses use argument arrays, bounded I/O, deadlines, and cancellation cleanup.
- Tests must not call live models unless explicitly opted in with GATEWAY_LIVE_PROFILE.
- Verify with `.venv/bin/pytest`, `.venv/bin/ruff check .`, and `.venv/bin/ruff format --check .`.
- Portable integration instructions live in src/local_llm_gateway/resources/INTEGRATION.md; keep root INTEGRATION.md synchronized.
