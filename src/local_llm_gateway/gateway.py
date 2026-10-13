from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from .config import GatewayConfig
from .contracts import GenerationRequest, GenerationResponse
from .errors import GatewayError
from .protocol import encode_request, validate_response
from .providers.base import Provider


class _Capacity:
    """Thread-safe capacity works for sync callers and multiple asyncio loops."""

    def __init__(self, active_limit: int, queue_limit: int):
        self.limit = active_limit
        self.total_limit = active_limit + queue_limit
        self.lock = threading.Lock()
        self.total = 0
        self.active = 0

    @asynccontextmanager
    async def slot(self):
        with self.lock:
            if self.total >= self.total_limit:
                raise GatewayError("queue_full", "Provider queue is full. Try again later.")
            self.total += 1
        acquired = False
        try:
            while not acquired:
                with self.lock:
                    if self.active < self.limit:
                        self.active += 1
                        acquired = True
                if not acquired:
                    await asyncio.sleep(0.01)
            yield
        finally:
            with self.lock:
                self.total -= 1
                if acquired:
                    self.active -= 1


class Gateway:
    def __init__(self, config: GatewayConfig, *, providers: dict[str, Provider] | None = None):
        self.config = config.model_copy(deep=True)
        if providers is None:
            from .providers.cli import CLIProvider

            providers = {
                name: CLIProvider(name, settings) for name, settings in config.providers.items()
            }
        self.providers = providers
        self._capacity = {
            name: _Capacity(
                config.runtime.max_concurrency_per_provider, config.runtime.max_queue_per_provider
            )
            for name in providers
        }

    @classmethod
    def from_config(cls, path: str | Path = "llm-gateway.yaml") -> Gateway:
        return cls(GatewayConfig.from_file(path))

    async def generate(
        self, request: GenerationRequest, *, profile: str | None = None
    ) -> GenerationResponse:
        started = time.monotonic()
        name = profile or self.config.default_profile
        if name not in self.config.profiles:
            raise GatewayError("unknown_profile", f"Unknown model profile: {name}")
        selected = self.config.profiles[name]
        provider = self.providers.get(selected.provider)
        if provider is None:
            raise GatewayError("provider_unavailable", "Configured provider is not available.")
        request = GenerationRequest.model_validate(request.model_dump())
        runtime = self.config.runtime
        if len(encode_request(request).encode()) > runtime.max_input_bytes:
            raise GatewayError(
                "input_too_large",
                "Request exceeds max_input_bytes; summarize history or reduce input.",
            )
        try:
            async with asyncio.timeout(runtime.timeout_seconds):
                async with self._capacity[selected.provider].slot():
                    result = await provider.generate(request, selected.model, runtime)
                    text, calls = validate_response(result.payload, request)
        except TimeoutError as exc:
            raise GatewayError(
                "timeout", "Request timed out, including queue time.", provider=selected.provider
            ) from exc
        return GenerationResponse(
            text=text,
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            provider=selected.provider,
            model=selected.model,
            reported_model=result.reported_model,
            usage=result.usage,
            duration_seconds=time.monotonic() - started,
        )

    def generate_sync(
        self, request: GenerationRequest, *, profile: str | None = None
    ) -> GenerationResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.generate(request, profile=profile))
        raise GatewayError(
            "async_context", "Use await gateway.generate(...) inside an active event loop."
        )
