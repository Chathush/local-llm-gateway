from dataclasses import dataclass
from typing import Any, Protocol

from ..config import RuntimeConfig
from ..contracts import GenerationRequest


@dataclass
class ProviderResult:
    payload: dict[str, Any]
    reported_model: str | None = None
    usage: dict[str, Any] | None = None


class Provider(Protocol):
    async def generate(
        self, request: GenerationRequest, model: str | None, runtime: RuntimeConfig
    ) -> ProviderResult: ...
