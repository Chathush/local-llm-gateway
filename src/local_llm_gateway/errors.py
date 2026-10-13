class GatewayError(Exception):
    """Stable, transport-independent error. Messages never contain raw CLI output."""

    def __init__(self, code: str, message: str, *, provider: str | None = None):
        super().__init__(message)
        self.code = code
        self.provider = provider

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "provider": self.provider}
