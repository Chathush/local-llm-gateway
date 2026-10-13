from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from ..config import ProviderConfig, RuntimeConfig
from ..contracts import GenerationRequest
from ..errors import GatewayError
from ..protocol import ENVELOPE_SCHEMA, SYSTEM_PROMPT, encode_request
from .base import ProviderResult
from .process import run_process

_CODEX_DISABLE = (
    "shell_tool",
    "unified_exec",
    "apps",
    "multi_agent",
    "hooks",
    "plugins",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "in_app_browser",
    "image_generation",
    "view_image",
    "code_mode",
    "code_mode_host",
    "memories",
    "skill_search",
    "skill_mcp_dependency_install",
    "goals",
    "sleep_tool",
)
_CONFLICTS = {
    "codex": (
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "OPENAI_BASE_URL",
        "CODEX_ACCESS_TOKEN",
        "CODEX_AUTH_TOKEN",
        "CODEX_WORKLOAD_IDENTITY_TOKEN_FILE",
    ),
    "claude_code": (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ),
}


def provider_error(raw: str, provider: str | None = None) -> GatewayError:
    value = raw.lower()
    if any(
        s in value
        for s in ("rate limit", "rate_limit", "usage limit", "quota", "429", "usage_limit")
    ):
        return GatewayError(
            "rate_limited",
            "Provider usage limit reached. Retry after the provider's reset window.",
            provider=provider,
        )
    if any(
        s in value
        for s in (
            "unauthorized",
            "not logged in",
            "authentication",
            "401",
            "login",
            "token expired",
        )
    ):
        return GatewayError(
            "authentication_required",
            "Provider authentication failed. Sign in again through the installed CLI.",
            provider=provider,
        )
    if "model" in value and any(
        s in value
        for s in (
            "not found",
            "not supported",
            "not available",
            "does not exist",
            "unsupported",
            "access",
        )
    ):
        return GatewayError(
            "model_unavailable",
            "Selected model is unavailable to this account/runtime. Change the configured model.",
            provider=provider,
        )
    return GatewayError(
        "provider_failed",
        "Provider command failed. Check CLI health, model access, and subscription usage with llm-gateway doctor.",
        provider=provider,
    )


def parse_codex(stdout: str) -> ProviderResult:
    try:
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        if not events or not all(isinstance(e, dict) for e in events):
            raise ValueError("Invalid events")
        last_text = None
        completed = None
        for event in events:
            if event.get("type") in {"error", "turn.failed"}:
                raise provider_error(json.dumps(event), "codex")
            item = event.get("item", {})
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                last_text = item.get("text")
            if item.get("type") in {
                "command_execution",
                "mcp_tool_call",
                "web_search",
                "file_change",
            }:
                raise GatewayError(
                    "unsupported_runtime",
                    "Codex attempted a runtime tool; model-only isolation was not honored.",
                    provider="codex",
                )
            if event.get("type") == "turn.completed":
                completed = event
        if completed is None or last_text is None:
            raise ValueError("No completed answer")
        payload = json.loads(last_text)
        if not isinstance(payload, dict):
            raise ValueError("Not an envelope")
        return ProviderResult(payload=payload, usage=completed.get("usage"))
    except GatewayError:
        raise
    except (ValueError, TypeError, AttributeError) as exc:
        raise GatewayError(
            "invalid_provider_response",
            "Codex did not return a completed JSON response.",
            provider="codex",
        ) from exc


def parse_claude(stdout: str) -> ProviderResult:
    try:
        data = json.loads(stdout)
        if not isinstance(data, dict) or data.get("type") != "result":
            raise ValueError("Missing result")
        if data.get("is_error") or data.get("subtype") != "success":
            raise provider_error(json.dumps(data), "claude_code")
        payload = data.get("structured_output")
        if payload is None:
            payload = json.loads(data.get("result", ""))
        if not isinstance(payload, dict):
            raise ValueError("Missing envelope")
        models = data.get("modelUsage", {})
        reported = next(iter(models)) if isinstance(models, dict) and len(models) == 1 else None
        return ProviderResult(payload=payload, usage=data.get("usage"), reported_model=reported)
    except GatewayError:
        raise
    except (ValueError, TypeError, AttributeError) as exc:
        raise GatewayError(
            "invalid_provider_response",
            "Claude Code did not return a completed JSON response.",
            provider="claude_code",
        ) from exc


class CLIProvider:
    def __init__(self, name: str, config: ProviderConfig):
        self.name = name
        self.config = config

    def environment(self) -> dict[str, str]:
        conflicts = [key for key in _CONFLICTS[self.name] if os.environ.get(key)]
        if conflicts:
            raise GatewayError(
                "auth_conflict",
                "Unset conflicting provider environment variables for this process: "
                + ", ".join(conflicts),
                provider=self.name,
            )
        allowed = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "TMPDIR",
            "TEMP",
            "TMP",
            "SystemRoot",
            "SYSTEMROOT",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "LANG",
            "LC_ALL",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "NODE_EXTRA_CA_CERTS",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "https_proxy",
            "http_proxy",
            "all_proxy",
            "no_proxy",
            "CODEX_HOME",
            "CLAUDE_CONFIG_DIR",
        }
        return {key: value for key, value in os.environ.items() if key in allowed}

    def executable(self) -> str:
        found = shutil.which(self.config.executable)
        if found is None:
            raise GatewayError(
                "provider_unavailable",
                f"Executable not found: {self.config.executable}",
                provider=self.name,
            )
        return str(Path(found).absolute())

    async def _check(self, executable, env, cwd, limit):
        async def run(args):
            return await run_process([executable, *args], cwd=cwd, env=env, limit=limit)

        help_result = await run(["exec", "--help"] if self.name == "codex" else ["--help"])
        needed = (
            ["--json", "--output-schema", "--ephemeral", "--ignore-user-config", "--ignore-rules"]
            if self.name == "codex"
            else [
                "--safe-mode",
                "--tools",
                "--strict-mcp-config",
                "--no-session-persistence",
                "--json-schema",
                "--permission-mode",
            ]
        )
        if help_result.returncode or any(flag not in help_result.stdout for flag in needed):
            raise GatewayError(
                "unsupported_runtime",
                "Installed CLI lacks required isolation/JSON flags. Upgrade it and rerun doctor.",
                provider=self.name,
            )
        if self.name == "codex":
            features = await run(["features", "list"])
            available = {line.split()[0] for line in features.stdout.splitlines() if line.strip()}
            required = {
                "shell_tool",
                "apps",
                "hooks",
                "plugins",
                "multi_agent",
                "skip_host_skill_discovery",
            }
            if features.returncode or not required.issubset(available):
                raise GatewayError(
                    "unsupported_runtime",
                    "Codex lacks required tool/customization controls. Upgrade Codex before using this gateway.",
                    provider=self.name,
                )
        status = await run(
            ["login", "status"]
            if self.name == "codex"
            else ["--safe-mode", "auth", "status", "--json"]
        )
        if self.name == "codex":
            authenticated = (
                status.returncode == 0
                and "logged in using chatgpt" in (status.stdout + status.stderr).lower()
            )
        else:
            try:
                info = json.loads(status.stdout)
                authenticated = (
                    status.returncode == 0
                    and info.get("loggedIn") is True
                    and info.get("authMethod") == "claude.ai"
                    and info.get("apiProvider") == "firstParty"
                )
            except (ValueError, AttributeError):
                authenticated = False
        if not authenticated:
            raise GatewayError(
                "authentication_required",
                "Subscription login required. Sign in through codex login or claude auth login; API/Console login is not accepted.",
                provider=self.name,
            )

    async def diagnose(self) -> dict:
        env = self.environment()
        executable = self.executable()
        with tempfile.TemporaryDirectory(prefix="llm-gateway-check-") as directory:
            cwd = Path(directory)
            async with asyncio.timeout(20):
                await self._check(executable, env, cwd, 1_000_000)
                version = await run_process([executable, "--version"], cwd=cwd, env=env, limit=4096)
        return {
            "provider": self.name,
            "executable": executable,
            "version": version.stdout.strip(),
            "subscription_login": True,
        }

    async def generate(
        self, request: GenerationRequest, model: str | None, runtime: RuntimeConfig
    ) -> ProviderResult:
        env = self.environment()
        executable = self.executable()
        with tempfile.TemporaryDirectory(prefix="llm-gateway-") as directory:
            cwd = Path(directory)
            await self._check(executable, env, cwd, runtime.max_output_bytes)
            if self.name == "codex":
                schema = cwd / "response.schema.json"
                schema.write_text(json.dumps(ENVELOPE_SCHEMA), encoding="utf-8")
                instructions = cwd / "instructions.txt"
                instructions.write_text(SYSTEM_PROMPT, encoding="utf-8")
                args = [
                    executable,
                    "exec",
                    "--json",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--sandbox",
                    "read-only",
                    "--output-schema",
                    str(schema),
                    "--color",
                    "never",
                ]
                settings = {
                    "model_provider": '"openai"',
                    "approval_policy": '"never"',
                    "web_search": '"disabled"',
                    "project_doc_max_bytes": "0",
                    "model_instructions_file": json.dumps(str(instructions)),
                    "features.skip_host_skill_discovery": "true",
                    **{f"features.{name}": "false" for name in _CODEX_DISABLE},
                }
                for key, value in settings.items():
                    args.extend(["-c", f"{key}={value}"])
                if model:
                    args.extend(["--model", model])
                args.append("-")
                prompt = "APPLICATION_REQUEST\n" + encode_request(request)
            else:
                args = [
                    executable,
                    "--safe-mode",
                    "--print",
                    "--output-format",
                    "json",
                    "--no-session-persistence",
                    "--tools",
                    "",
                    "--strict-mcp-config",
                    "--mcp-config",
                    '{"mcpServers":{}}',
                    "--permission-mode",
                    "dontAsk",
                    "--system-prompt",
                    SYSTEM_PROMPT,
                    "--json-schema",
                    json.dumps(ENVELOPE_SCHEMA),
                ]
                if model:
                    args.extend(["--model", model])
                prompt = "APPLICATION_REQUEST\n" + encode_request(request)
            result = await run_process(
                args, stdin=prompt, cwd=cwd, env=env, limit=runtime.max_output_bytes
            )
            if result.returncode:
                raise provider_error(result.stderr + result.stdout, self.name)
            return (
                parse_codex(result.stdout) if self.name == "codex" else parse_claude(result.stdout)
            )
