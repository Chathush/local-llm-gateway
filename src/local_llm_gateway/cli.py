import argparse
import asyncio
import json
import os
import secrets
import sys
from importlib.resources import files
from pathlib import Path

from pydantic import ValidationError
from yaml import YAMLError

from .config import GatewayConfig
from .contracts import GenerationRequest, Message, Tool
from .errors import GatewayError
from .gateway import Gateway
from .providers.cli import CLIProvider


async def doctor(config):
    reports = []
    for name in sorted({p.provider for p in config.profiles.values()}):
        try:
            reports.append(
                {"ok": True, **await CLIProvider(name, config.providers[name]).diagnose()}
            )
        except GatewayError as exc:
            reports.append({"ok": False, "provider": name, "error": exc.as_dict()})
        except TimeoutError:
            reports.append(
                {
                    "ok": False,
                    "provider": name,
                    "error": {"code": "timeout", "message": "CLI health check timed out."},
                }
            )
    print(
        json.dumps(
            {
                "providers": reports,
                "note": "Login/CLI checks only. Use smoke to verify selected-model inference.",
            },
            indent=2,
        )
    )
    return 0 if all(r["ok"] for r in reports) else 1


async def smoke(gateway, profile, with_tool):
    if not with_tool:
        result = await gateway.generate(
            GenerationRequest(
                messages=[Message(role="user", content="Reply with exactly gateway-ok.")]
            ),
            profile=profile,
        )
    else:
        request = GenerationRequest(
            messages=[
                Message(
                    role="user",
                    content="Use add to calculate 2 + 3, then report the numeric answer.",
                )
            ],
            tools=[
                Tool(
                    name="add",
                    description="Add two integers",
                    parameters={
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                )
            ],
            tool_choice="add",
        )
        first = await gateway.generate(request, profile=profile)
        messages = [*request.messages, first.as_message()]
        for call in first.tool_calls:
            messages.append(
                Message(
                    role="tool",
                    content=str(call.arguments["a"] + call.arguments["b"]),
                    tool_call_id=call.id,
                )
            )
        result = await gateway.generate(
            GenerationRequest(messages=messages, tools=request.tools, tool_choice="none"),
            profile=profile,
        )
    print(result.model_dump_json(indent=2))
    return 0


def init(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    resources = files("local_llm_gateway").joinpath("resources")
    for source, target in [
        ("example.yaml", "llm-gateway.yaml"),
        ("INTEGRATION.md", "LLM_GATEWAY_INTEGRATION.md"),
    ]:
        destination = directory / target
        try:
            with destination.open("x", encoding="utf-8") as output:
                output.write(resources.joinpath(source).read_text(encoding="utf-8"))
            print(f"Created {destination}")
        except FileExistsError:
            print(f"Kept existing {destination}")
    print(
        "Add a reference to LLM_GATEWAY_INTEGRATION.md in the application's AGENTS.md or CLAUDE.md."
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local subscription-backed model gateway")
    sub = parser.add_subparsers(dest="command", required=True)
    initialize = sub.add_parser(
        "init", help="Copy config and integration guide without overwriting"
    )
    initialize.add_argument("directory", nargs="?", default=".")
    for command in ("doctor", "models", "smoke", "serve"):
        child = sub.add_parser(command)
        child.add_argument("--config", default="llm-gateway.yaml")
        if command == "smoke":
            child.add_argument("--profile")
            child.add_argument(
                "--tool",
                action="store_true",
                help="Verify a complete tool round trip (two model calls)",
            )
        elif command == "serve":
            child.add_argument("--host", choices=["127.0.0.1", "::1"], default="127.0.0.1")
            child.add_argument("--port", type=int, default=8787)
            child.add_argument("--token-file", default=".gateway-token")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return init(args.directory)
        config = GatewayConfig.from_file(args.config)
        if args.command == "doctor":
            return asyncio.run(doctor(config))
        if args.command == "models":
            print(
                json.dumps(
                    {
                        "profiles": {name: p.model_dump() for name, p in config.profiles.items()},
                        "note": "Configured aliases, not an entitlement catalog.",
                    },
                    indent=2,
                )
            )
            return 0
        gateway = Gateway(config)
        if args.command == "smoke":
            return asyncio.run(smoke(gateway, args.profile, args.tool))
        if args.command == "serve":
            import uvicorn

            from .server import create_app

            path = Path(args.token_file)
            if not path.exists():
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w") as output:
                    output.write(secrets.token_urlsafe(32) + "\n")
            token = path.read_text().strip()
            print(f"Local access token: {path.absolute()} (keep private)")
            uvicorn.run(
                create_app(gateway, token=token), host=args.host, port=args.port, access_log=False
            )
            return 0
    except GatewayError as exc:
        print(json.dumps({"error": exc.as_dict()}), file=sys.stderr)
        return 1
    except (OSError, ValidationError, ValueError, YAMLError) as exc:
        print(
            f"Configuration/setup error ({type(exc).__name__}). Check file paths and configuration schema.",
            file=sys.stderr,
        )
        return 2
    except ImportError:
        print(
            "Install the required optional dependencies, e.g. pip install 'local-llm-gateway[http]'.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
