"""PostgreSQL-backed FastAPI composition without company algorithm code."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Mapping

from fastapi import FastAPI

from puncture_agent.runtime import InMemoryRunService
from puncture_agent.runtime.postgres_repository import PostgresRunRepository
from puncture_agent.runtime.service import RunExecutor

from .fastapi_app import (
    ArtifactAccessGateway,
    HealthProbe,
    PrincipalAuthenticator,
    ResourceAuthorizer,
    create_app,
)
from .http_metrics import HttpMetrics


def _parse_positive_int(name: str, value: str) -> int:
    if not value or not value.isdigit() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _parse_positive_float(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be positive and finite") from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return parsed


@dataclass(frozen=True, slots=True)
class PostgresApiSettings:
    connection_string: str = field(repr=False)
    schema: str = "puncture_runtime"
    connect_timeout_seconds: float = 5.0
    statement_timeout_ms: int = 5_000
    lock_timeout_ms: int = 1_000
    max_request_body_bytes: int = 1024 * 1024
    migrate_on_startup: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.connection_string, str)
            or not self.connection_string.strip()
        ):
            raise ValueError("connection_string is required")
        if (
            isinstance(self.max_request_body_bytes, bool)
            or not isinstance(self.max_request_body_bytes, int)
            or self.max_request_body_bytes < 1
            or self.max_request_body_bytes > 64 * 1024 * 1024
        ):
            raise ValueError(
                "max_request_body_bytes must be between 1 byte and 64 MiB"
            )
        if not isinstance(self.migrate_on_startup, bool):
            raise TypeError("migrate_on_startup must be a boolean")

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "PostgresApiSettings":
        source = os.environ if env is None else env
        try:
            connection_string = source["PUNCTURE_API_POSTGRES_DSN"]
        except KeyError as exc:
            raise ValueError("PUNCTURE_API_POSTGRES_DSN is required") from exc
        return cls(
            connection_string=connection_string,
            schema=source.get("PUNCTURE_API_POSTGRES_SCHEMA", "puncture_runtime"),
            connect_timeout_seconds=_parse_positive_float(
                "PUNCTURE_API_POSTGRES_CONNECT_TIMEOUT_SECONDS",
                source.get("PUNCTURE_API_POSTGRES_CONNECT_TIMEOUT_SECONDS", "5"),
            ),
            statement_timeout_ms=_parse_positive_int(
                "PUNCTURE_API_POSTGRES_STATEMENT_TIMEOUT_MS",
                source.get("PUNCTURE_API_POSTGRES_STATEMENT_TIMEOUT_MS", "5000"),
            ),
            lock_timeout_ms=_parse_positive_int(
                "PUNCTURE_API_POSTGRES_LOCK_TIMEOUT_MS",
                source.get("PUNCTURE_API_POSTGRES_LOCK_TIMEOUT_MS", "1000"),
            ),
            max_request_body_bytes=_parse_positive_int(
                "PUNCTURE_API_MAX_REQUEST_BODY_BYTES",
                source.get("PUNCTURE_API_MAX_REQUEST_BODY_BYTES", str(1024 * 1024)),
            ),
            migrate_on_startup=_parse_bool(
                "PUNCTURE_API_MIGRATE_ON_STARTUP",
                source.get("PUNCTURE_API_MIGRATE_ON_STARTUP", "false"),
            ),
        )


class _PostgresHealthProbe:
    def __init__(
        self,
        repository: PostgresRunRepository,
        optional_probe: HealthProbe | None,
        *,
        artifact_gateway_configured: bool,
    ) -> None:
        self._repository = repository
        self._optional_probe = optional_probe
        self._artifact_gateway_configured = artifact_gateway_configured

    def status(self) -> str:
        self._repository.check_health()
        status = "UP" if self._artifact_gateway_configured else "DEGRADED"
        if self._optional_probe is None:
            return status
        try:
            optional_status = self._optional_probe.status()
            if optional_status == "DEGRADED":
                return "DEGRADED"
            return status
        except Exception:
            return "DEGRADED"


def create_postgres_app(
    settings: PostgresApiSettings,
    *,
    executor: RunExecutor,
    authenticator: PrincipalAuthenticator,
    authorizer: ResourceAuthorizer,
    artifact_gateway: ArtifactAccessGateway | None = None,
    optional_health_probe: HealthProbe | None = None,
    metrics: HttpMetrics | None = None,
) -> FastAPI:
    """Compose FastAPI with PostgreSQL Run persistence and injected execution.

    ``executor`` is intentionally mandatory.  The repository contains no
    company algorithm implementation, and callers may inject their production
    executor when it becomes available.
    """

    if not isinstance(settings, PostgresApiSettings):
        raise TypeError("settings must be PostgresApiSettings")
    if executor is None:
        raise TypeError("executor is required")
    repository = PostgresRunRepository(
        settings.connection_string,
        schema=settings.schema,
        connect_timeout_seconds=settings.connect_timeout_seconds,
        statement_timeout_ms=settings.statement_timeout_ms,
        lock_timeout_ms=settings.lock_timeout_ms,
    )
    service = InMemoryRunService(executor, repository=repository)
    app = create_app(
        service,
        authenticator=authenticator,
        authorizer=authorizer,
        artifact_gateway=artifact_gateway,
        health_probe=_PostgresHealthProbe(
            repository,
            optional_health_probe,
            artifact_gateway_configured=artifact_gateway is not None,
        ),
        metrics=metrics,
        max_request_body_bytes=settings.max_request_body_bytes,
        startup_hooks=(repository.migrate,) if settings.migrate_on_startup else (),
    )
    app.state.run_repository = repository
    app.state.run_service = service
    return app


__all__ = ["PostgresApiSettings", "create_postgres_app"]
