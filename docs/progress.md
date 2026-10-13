# Progress: docs/implementation-plan.md

- Approved design and implementation scope recorded. Empty user-specified folder; work stays here.
- Execute inline using test-first development. No Git worktree applies because this folder is not a Git repository.
- Interfaces: providers return provider results; core validates response envelope; adapters only translate the public contract.

- Tasks 1–5 implemented and verified: core, subprocess providers, framework adapters, HTTP/CLI, docs/examples/distribution.
- Ruling: execute directly in the user-requested empty folder; no Git/worktree or publication operations were needed.
- Ruling: use ADK >=2.11 because the adapter uses its capability interface; older versions are not claimed.
- Final review fix: schema-aware ADK JSON Schema conversion; regression failed before fix, then full suite passed.
- Final evidence: 45 tests passed, 1 opt-in live test skipped by default; live Codex tool/JSON test passed separately. Ruff, pip check, package build, and clean-environment wheel install passed.
- External verification limits: Claude login expired; subsequent live framework examples hit Codex usage limits. See verification.md.
- Reviewer exclusions resolved as documented limits: no OS isolation guarantee, no all-version compatibility claim, no successful live Claude claim. No deferred code-review minors.
