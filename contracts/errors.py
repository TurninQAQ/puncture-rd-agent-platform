"""Structured tool errors.  Error strings alone are not a stable API."""

from dataclasses import dataclass, field

from .enums import ErrorCode


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: ErrorCode
    message: str
    retryable: bool
    field_path: str | None = None
    dependency: str | None = None
    details: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("error message must not be empty")
        object.__setattr__(self, "details", dict(self.details))
