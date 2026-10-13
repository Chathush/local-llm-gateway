"""Authenticated local HTTP transport with an explicit Chat Completions subset."""

import asyncio
import hmac
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .contracts import GenerationRequest, Message, Tool, ToolCall
from .errors import GatewayError
from .gateway import Gateway

_STATUS = {
    "authentication_required": 503,
    "auth_conflict": 503,
    "provider_unavailable": 503,
    "unsupported_runtime": 503,
    "queue_full": 429,
    "rate_limited": 429,
    "timeout": 504,
    "provider_failed": 502,
    "invalid_provider_response": 502,
    "output_too_large": 502,
    "input_too_large": 413,
    "client_disconnected": 499,
}


def error_response(error, status=None):
    return JSONResponse(
        {"error": {**error.as_dict(), "type": "gateway_error"}},
        status_code=status or _STATUS.get(error.code, 400),
    )


class LocalBoundary:
    def __init__(self, app, token, max_bytes):
        self.app, self.token, self.max_bytes = app, token.encode(), max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if scope["path"] != "/health":
            provided = headers.get(b"authorization", b"")
            if not hmac.compare_digest(provided, b"Bearer " + self.token):
                response = error_response(
                    GatewayError("unauthorized", "A valid local gateway bearer token is required."),
                    401,
                )
                return await response(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                response = error_response(
                    GatewayError(
                        "input_too_large", "HTTP request body exceeds the configured size limit."
                    ),
                    413,
                )
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        replayed = False

        async def bounded_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return await self.app(scope, bounded_receive, send)


def only_fields(value, allowed):
    if not isinstance(value, dict):
        raise GatewayError("invalid_request", "Expected a JSON object.")
    if set(value) - set(allowed):
        raise GatewayError(
            "unsupported_capability",
            "Request contains unsupported fields. See the gateway's HTTP compatibility documentation.",
        )


def text_content(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            only_fields(item, {"type", "text"})
            if item.get("type") != "text" or not isinstance(item.get("text"), str):
                raise GatewayError(
                    "unsupported_capability", "Only text content blocks are supported."
                )
            parts.append(item["text"])
        return "\n".join(parts)
    raise GatewayError("invalid_request", "Invalid message content.")


def chat_request(data):
    only_fields(
        data, {"model", "messages", "tools", "tool_choice", "response_format", "stream", "n"}
    )
    if data.get("stream", False) is not False or data.get("n", 1) != 1:
        raise GatewayError("unsupported_capability", "Only stream=false and n=1 are supported.")
    if not isinstance(data.get("model"), str) or not data["model"]:
        raise GatewayError("invalid_request", "model must be a configured profile name.")
    messages = []
    for raw in data["messages"]:
        only_fields(raw, {"role", "content", "name", "tool_call_id", "tool_calls"})
        calls = []
        for call in raw.get("tool_calls", []):
            only_fields(call, {"id", "type", "function"})
            if call.get("type") != "function":
                raise GatewayError(
                    "unsupported_capability", "Only function tool calls are supported."
                )
            fn = call["function"]
            only_fields(fn, {"name", "arguments"})
            calls.append(
                ToolCall(id=call["id"], name=fn["name"], arguments=json.loads(fn["arguments"]))
            )
        messages.append(
            Message(
                role=raw["role"],
                content=text_content(raw.get("content")),
                name=raw.get("name"),
                tool_call_id=raw.get("tool_call_id"),
                tool_calls=calls,
            )
        )
    tools = []
    for raw in data.get("tools", []):
        only_fields(raw, {"type", "function"})
        if raw.get("type") != "function":
            raise GatewayError("unsupported_capability", "Only function tools are supported.")
        fn = raw["function"]
        only_fields(fn, {"name", "description", "parameters", "strict"})
        tools.append(
            Tool(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {"type": "object"}),
            )
        )
    choice = data.get("tool_choice", "auto")
    if isinstance(choice, dict):
        only_fields(choice, {"type", "function"})
        if choice.get("type") != "function":
            raise GatewayError("unsupported_capability", "Unsupported tool_choice.")
        only_fields(choice["function"], {"name"})
        choice = choice["function"]["name"]
    schema = None
    fmt = data.get("response_format")
    if fmt is not None:
        only_fields(fmt, {"type", "json_schema"})
        if fmt.get("type") == "json_object":
            schema = {"type": "object"}
        elif fmt.get("type") == "json_schema":
            descriptor = fmt["json_schema"]
            only_fields(descriptor, {"name", "description", "schema", "strict"})
            schema = descriptor["schema"]
        elif fmt.get("type") != "text":
            raise GatewayError("unsupported_capability", "Unsupported response_format.")
    return GenerationRequest(
        messages=messages, tools=tools, tool_choice=choice, output_schema=schema
    )


async def while_connected(request, operation):
    async def disconnected():
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    work = asyncio.create_task(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return await work
        raise GatewayError("client_disconnected", "Client disconnected.")
    finally:
        for task in (work, watcher):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)


def create_app(gateway: Gateway, *, token: str) -> FastAPI:
    if not token or token.strip() != token:
        raise ValueError("A nonempty local bearer token is required")
    app = FastAPI(
        title="Local LLM Gateway", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.add_middleware(LocalBoundary, token=token, max_bytes=gateway.config.runtime.max_input_bytes)

    @app.exception_handler(GatewayError)
    async def handle_error(request, exc):
        return error_response(exc)

    async def body(request):
        try:
            value = await request.json()
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (ValueError, TypeError) as exc:
            raise GatewayError(
                "invalid_request", "Request must contain a valid JSON object."
            ) from exc

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": p.provider}
                for name, p in gateway.config.profiles.items()
            ],
        }

    @app.post("/generate")
    async def generate(request: Request):
        data = await body(request)
        try:
            only_fields(data, {"profile", "messages", "tools", "tool_choice", "output_schema"})
            profile = data.pop("profile", None)
            if profile is not None and not isinstance(profile, str):
                raise ValueError()
            req = GenerationRequest.model_validate(data)
        except (ValidationError, ValueError, TypeError) as exc:
            raise GatewayError(
                "invalid_request", "Invalid generation request; check roles, tool IDs, and schemas."
            ) from exc
        result = await while_connected(request, gateway.generate(req, profile=profile))
        return result.model_dump()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        data = await body(request)
        try:
            req = chat_request(data)
        except (ValidationError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise GatewayError(
                "invalid_request", "Invalid chat request; check roles, tool IDs, and schemas."
            ) from exc
        result = await while_connected(request, gateway.generate(req, profile=data["model"]))
        message = {"role": "assistant", "content": result.text or None}
        if result.tool_calls:
            message["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in result.tool_calls
            ]
        usage = None
        if result.usage and all(
            isinstance(result.usage.get(k), int) for k in ("input_tokens", "output_tokens")
        ):
            prompt = (
                result.usage["input_tokens"]
                + (result.usage.get("cache_read_input_tokens", 0) or 0)
                + (result.usage.get("cache_creation_input_tokens", 0) or 0)
            )
            completion = result.usage["output_tokens"]
            usage = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        return {
            "id": result.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason}],
            "usage": usage,
        }

    return app
