from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from .contracts import StrictModel


class Profile(StrictModel):
    provider: Literal["codex", "claude_code"]
    model: str | None = Field(default=None, min_length=1)


class ProviderConfig(StrictModel):
    executable: str = Field(min_length=1)
    auth: Literal["existing_cli_subscription"] = "existing_cli_subscription"


class RuntimeConfig(StrictModel):
    timeout_seconds: float = Field(default=120, gt=0, le=3600)
    max_concurrency_per_provider: int = Field(default=1, ge=1, le=32)
    max_queue_per_provider: int = Field(default=8, ge=0, le=1024)
    max_input_bytes: int = Field(default=1_000_000, ge=1024, le=20_000_000)
    max_output_bytes: int = Field(default=4_000_000, ge=1024, le=100_000_000)


class GatewayConfig(StrictModel):
    schema_version: Literal[1] = 1
    application: str = "local-app"
    default_profile: str = "primary"
    profiles: dict[str, Profile] = Field(min_length=1)
    providers: dict[str, ProviderConfig] = Field(
        default_factory=lambda: {
            "codex": ProviderConfig(executable="codex"),
            "claude_code": ProviderConfig(executable="claude"),
        }
    )
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def valid_profiles(self):
        if self.default_profile not in self.profiles:
            raise ValueError("default_profile must name a configured profile")
        for name, profile in self.profiles.items():
            if not name or profile.provider not in self.providers:
                raise ValueError("Each profile needs a name and a configured provider")
        if set(self.providers) - {"codex", "claude_code"}:
            raise ValueError("Unknown provider configuration")
        return self

    @classmethod
    def from_file(cls, path: str | Path):
        with Path(path).open(encoding="utf-8") as file:
            return cls.model_validate(yaml.safe_load(file))
