"""PostgreSQL-backed FastAPI composition without company algorithm code."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Mapping
from uuid import uuid4

from fastapi import FastAPI

from puncture_agent.runtime import InMemoryRunService
from puncture_agent.runtime.postgres_repository import PostgresRunRepository
from puncture_agent.runtime.service import RecoverableRunExecutor, RunExecutor
from puncture_agent.runtime.worker import (
    RunWorker,
    WorkerConfig,
    WorkerState,
)

from .fastapi_app import (
    ArtifactAccessGateway,
    HealthProbe,
    PrincipalAuthenticator,
    ResourceAuthorizer,
    create_app,
)
from .http_metrics import HttpMetrics
from .sse import SseConfig


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
    sse_page_size: int = 128
    sse_poll_interval_seconds: float = 1.0
    sse_heartbeat_seconds: float = 15.0
    sse_max_connection_seconds: float = 600.0
    sse_max_connections: int = 200
    sse_max_connections_per_tenant: int = 20
    worker_enabled: bool = True
    worker_id: str | None = None
    worker_concurrency: int = 1
    worker_poll_interval_seconds: float = 0.5
    worker_heartbeat_interval_seconds: float = 5.0
    worker_lease_seconds: float = 30.0
    worker_shutdown_grace_seconds: float = 30.0

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
        if not isinstance(self.worker_enabled, bool):
            raise TypeError("worker_enabled must be a boolean")
        self.to_sse_config()
        self.to_worker_config(worker_id=self.worker_id or "worker-validation")

    def to_sse_config(self) -> SseConfig:
        return SseConfig(
            page_size=self.sse_page_size,
            poll_interval_seconds=self.sse_poll_interval_seconds,
            heartbeat_seconds=self.sse_heartbeat_seconds,
            max_connection_seconds=self.sse_max_connection_seconds,
            max_connections=self.sse_max_connections,
            max_connections_per_tenant=self.sse_max_connections_per_tenant,
        )

    def to_worker_config(self, *, worker_id: str | None = None) -> WorkerConfig:
        resolved_worker_id = (
            worker_id if worker_id is not None else self.worker_id
        )
        if resolved_worker_id is None:
            raise ValueError("worker_id is required when building WorkerConfig")
        return WorkerConfig(
            worker_id=resolved_worker_id,
            concurrency=self.worker_concurrency,
            poll_interval_seconds=self.worker_poll_interval_seconds,
            heartbeat_interval_seconds=self.worker_heartbeat_interval_seconds,
            lease_seconds=self.worker_lease_seconds,
            shutdown_grace_seconds=self.worker_shutdown_grace_seconds,
        )

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
            sse_page_size=_parse_positive_int(
                "PUNCTURE_API_SSE_PAGE_SIZE",
                source.get("PUNCTURE_API_SSE_PAGE_SIZE", "128"),
            ),
            sse_poll_interval_seconds=_parse_positive_float(
                "PUNCTURE_API_SSE_POLL_INTERVAL_SECONDS",
                source.get("PUNCTURE_API_SSE_POLL_INTERVAL_SECONDS", "1"),
            ),
            sse_heartbeat_seconds=_parse_positive_float(
                "PUNCTURE_API_SSE_HEARTBEAT_SECONDS",
                source.get("PUNCTURE_API_SSE_HEARTBEAT_SECONDS", "15"),
            ),
            sse_max_connection_seconds=_parse_positive_float(
                "PUNCTURE_API_SSE_MAX_CONNECTION_SECONDS",
                source.get("PUNCTURE_API_SSE_MAX_CONNECTION_SECONDS", "600"),
            ),
            sse_max_connections=_parse_positive_int(
                "PUNCTURE_API_SSE_MAX_CONNECTIONS",
                source.get("PUNCTURE_API_SSE_MAX_CONNECTIONS", "200"),
            ),
            sse_max_connections_per_tenant=_parse_positive_int(
                "PUNCTURE_API_SSE_MAX_CONNECTIONS_PER_TENANT",
                source.get(
                    "PUNCTURE_API_SSE_MAX_CONNECTIONS_PER_TENANT",
                    "20",
                ),
            ),
            worker_enabled=_parse_bool(
                "PUNCTURE_API_WORKER_ENABLED",
                source.get("PUNCTURE_API_WORKER_ENABLED", "true"),
            ),
            worker_id=source.get("PUNCTURE_API_WORKER_ID") or None,
            worker_concurrency=_parse_positive_int(
                "PUNCTURE_API_WORKER_CONCURRENCY",
                source.get("PUNCTURE_API_WORKER_CONCURRENCY", "1"),
            ),
            worker_poll_interval_seconds=_parse_positive_float(
                "PUNCTURE_API_WORKER_POLL_INTERVAL_SECONDS",
                source.get("PUNCTURE_API_WORKER_POLL_INTERVAL_SECONDS", "0.5"),
            ),
            worker_heartbeat_interval_seconds=_parse_positive_float(
                "PUNCTURE_API_WORKER_HEARTBEAT_INTERVAL_SECONDS",
                source.get(
                    "PUNCTURE_API_WORKER_HEARTBEAT_INTERVAL_SECONDS",
                    "5",
                ),
            ),
            worker_lease_seconds=_parse_positive_float(
                "PUNCTURE_API_WORKER_LEASE_SECONDS",
                source.get("PUNCTURE_API_WORKER_LEASE_SECONDS", "30"),
            ),
            worker_shutdown_grace_seconds=_parse_positive_float(
                "PUNCTURE_API_WORKER_SHUTDOWN_GRACE_SECONDS",
                source.get(
                    "PUNCTURE_API_WORKER_SHUTDOWN_GRACE_SECONDS",
                    "30",
                ),
            ),
        )


class _PostgresHealthProbe:
    def __init__(
        self,
        repository: PostgresRunRepository,
        optional_probe: HealthProbe | None,
        *,
        artifact_gateway_configured: bool,
        worker: RunWorker | None,
    ) -> None:
        self._repository = repository
        self._optional_probe = optional_probe
        self._artifact_gateway_configured = artifact_gateway_configured
        self._worker = worker

    def status(self) -> str:
        self._repository.check_health()
        if (
            self._worker is not None
            and self._worker.status.state is not WorkerState.RUNNING
        ):
            raise RuntimeError("configured execution worker is not running")
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
    executor: RunExecutor | RecoverableRunExecutor,
    authenticator: PrincipalAuthenticator,
    authorizer: ResourceAuthorizer,
    artifact_gateway: ArtifactAccessGateway | None = None,
    optional_health_probe: HealthProbe | None = None,
    metrics: HttpMetrics | None = None,
) -> FastAPI:
    """Compose FastAPI with PostgreSQL Run persistence and injected execution.

    ``executor`` is intentionally mandatory. The repository contains no
    company algorithm implementation. Worker mode requires the injected
    executor to implement the recovery-safe claimed-execution port.
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
    worker: RunWorker | None = None
    if settings.worker_enabled:
        wakeup_event = RunWorker.create_wakeup_event()
        worker_id = settings.worker_id or f"api-{os.getpid()}-{uuid4().hex}"
        service = InMemoryRunService(
            executor,
            repository=repository,
            deferred_execution=True,
            execution_notifier=wakeup_event.set,
        )
        worker = RunWorker(
            repository,
            service,
            config=settings.to_worker_config(worker_id=worker_id),
            wakeup_event=wakeup_event,
        )
    else:
        service = InMemoryRunService(executor, repository=repository)

    startup_hooks = []
    if settings.migrate_on_startup:
        startup_hooks.append(repository.migrate)
    if worker is not None:
        startup_hooks.append(worker.start)
    app = create_app(
        service,
        authenticator=authenticator,
        authorizer=authorizer,
        artifact_gateway=artifact_gateway,
        health_probe=_PostgresHealthProbe(
            repository,
            optional_health_probe,
            artifact_gateway_configured=artifact_gateway is not None,
            worker=worker,
        ),
        metrics=metrics,
        sse_config=settings.to_sse_config(),
        additional_metrics=(worker.metrics,) if worker is not None else (),
        max_request_body_bytes=settings.max_request_body_bytes,
        startup_hooks=tuple(startup_hooks),
        shutdown_hooks=(worker.stop,) if worker is not None else (),
    )
    app.state.run_repository = repository
    app.state.run_service = service
    app.state.run_worker = worker
    return app


__all__ = ["PostgresApiSettings", "create_postgres_app"]
