"""FastAPI transport for the fixed Run contracts.

Authentication, project/case authorization, artifact metadata access, and the
Run service are explicit injected ports.  The transport never calls a medical
algorithm directly and has no permissive default principal.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Any, Callable, Literal, Protocol, Sequence

from fastapi import Depends, FastAPI, Path, Query, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHttpException

from contracts.artifacts import ArtifactPublicView
from contracts.enums import ArtifactStatus
from puncture_agent.artifacts.registry import ArtifactRegistryError
from puncture_agent.runtime import (
    ApprovalDecision,
    RunEvent,
    RunEventPage,
    RunRequest,
    RunSnapshot,
)
from puncture_agent.runtime.errors import RunServiceError

from .body_admission import RawBodyAdmissionMiddleware
from .http_contracts import (
    ApiErrorDetail,
    ApiErrorResponse,
    ApiRequestValidationError,
    ApprovalBody,
    ArtifactMetadataResponse,
    AuthenticatedPrincipal,
    HealthResponse,
    HttpErrorMapping,
    RunCreateBody,
    RunEventResponse,
    RunSnapshotResponse,
    map_exception_to_http_error,
)
from .http_metrics import HttpMetrics, HttpMetricsMiddleware
from .privacy import PublicValueValidationError
from .sse import (
    SseConfig,
    SseConnectionLimiter,
    SseMetrics,
    SseNegotiationError,
    SseStreamingResponse,
    encode_event_page,
    negotiate_event_representation,
    resolve_event_cursor,
    stream_event_pages,
)


_RESOURCE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$"
_ARTIFACT_RESOURCE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"
_AUTHORIZED_IDENTIFIER = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$"
)
_BEARER_TOKEN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z:/=]+$")
_MAX_BEARER_TOKEN_BYTES = 8192


class ApiPermission(str, Enum):
    RUN_CREATE = "run:create"
    RUN_READ = "run:read"
    RUN_EVENTS_READ = "run:events:read"
    RUN_APPROVE = "run:approve"
    RUN_CANCEL = "run:cancel"
    RUN_RESUME = "run:resume"
    ARTIFACT_USE = "artifact:use"
    ARTIFACT_READ = "artifact:read"


@dataclass(frozen=True, slots=True)
class AuthorizedCase:
    """Authoritative server-side tenant/project/case resolution."""

    tenant_id: str
    project_id: str
    case_id: str

    def __post_init__(self) -> None:
        for field_name in ("tenant_id", "project_id", "case_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} is required")
            normalized = value.strip()
            if not _AUTHORIZED_IDENTIFIER.fullmatch(normalized):
                raise ValueError(f"{field_name} contains unsupported characters")
            object.__setattr__(self, field_name, normalized)


class PrincipalAuthenticator(Protocol):
    """Verify a bearer token and return trusted identity context."""

    def authenticate(self, bearer_token: str) -> AuthenticatedPrincipal: ...


class ResourceAuthorizer(Protocol):
    """Resolve project/case ownership and authorize every Run operation."""

    def require_case(
        self,
        principal: AuthenticatedPrincipal,
        *,
        case_id: str,
        permission: ApiPermission,
    ) -> AuthorizedCase: ...

    def require_run(
        self,
        principal: AuthenticatedPrincipal,
        *,
        snapshot: RunSnapshot,
        permission: ApiPermission,
    ) -> AuthorizedCase: ...


class ArtifactAccessGateway(Protocol):
    """Atomic artifact lookup and authorization boundary.

    A production implementation owns the private registry/IAM lookup.  Only
    ``ArtifactPublicView`` values may cross this port.
    """

    def require_artifacts(
        self,
        principal: AuthenticatedPrincipal,
        *,
        case: AuthorizedCase,
        artifact_ids: tuple[str, ...],
        permission: ApiPermission,
    ) -> tuple[ArtifactPublicView, ...]: ...

    def get_metadata(
        self,
        principal: AuthenticatedPrincipal,
        *,
        artifact_id: str,
        permission: ApiPermission,
    ) -> ArtifactPublicView: ...


class RunService(Protocol):
    def create_run(self, request: RunRequest) -> RunSnapshot: ...

    def get_run(self, run_id: str, *, tenant_id: str) -> RunSnapshot: ...

    def get_events(
        self,
        run_id: str,
        *,
        tenant_id: str,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]: ...

    def get_event_page(
        self,
        run_id: str,
        *,
        tenant_id: str,
        after_sequence: int = 0,
        limit: int = 128,
    ) -> RunEventPage: ...

    def approve(
        self,
        run_id: str,
        decision: ApprovalDecision,
        *,
        tenant_id: str,
    ) -> RunSnapshot: ...

    def cancel(self, run_id: str, *, tenant_id: str) -> RunSnapshot: ...

    def resume(self, run_id: str, *, tenant_id: str) -> RunSnapshot: ...


class HealthProbe(Protocol):
    def status(self) -> Literal["UP", "DEGRADED"]: ...


class AlwaysUpHealthProbe:
    def status(self) -> Literal["UP"]:
        return "UP"


class _DegradedHealthProbe:
    def status(self) -> Literal["DEGRADED"]:
        return "DEGRADED"


class _UnavailableArtifactGateway:
    def require_artifacts(
        self,
        principal: AuthenticatedPrincipal,
        *,
        case: AuthorizedCase,
        artifact_ids: tuple[str, ...],
        permission: ApiPermission,
    ) -> tuple[ArtifactPublicView, ...]:
        del principal, case, artifact_ids, permission
        raise RunServiceError(
            "ARTIFACT_GATEWAY_UNAVAILABLE",
            "artifact gateway is not configured",
            retryable=True,
        )

    def get_metadata(
        self,
        principal: AuthenticatedPrincipal,
        *,
        artifact_id: str,
        permission: ApiPermission,
    ) -> ArtifactPublicView:
        del principal, artifact_id, permission
        raise RunServiceError(
            "ARTIFACT_GATEWAY_UNAVAILABLE",
            "artifact gateway is not configured",
            retryable=True,
        )


@dataclass(frozen=True, slots=True)
class _Dependencies:
    run_service: RunService
    authenticator: PrincipalAuthenticator
    authorizer: ResourceAuthorizer
    artifact_gateway: ArtifactAccessGateway
    health_probe: HealthProbe
    allow_test_controls: bool


def _error_responses() -> dict[int | str, dict[str, Any]]:
    return {
        400: {"model": ApiErrorResponse},
        403: {"model": ApiErrorResponse},
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        413: {"model": ApiErrorResponse},
        415: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
        503: {"model": ApiErrorResponse},
    }


def _event_responses() -> dict[int | str, dict[str, Any]]:
    responses = _error_responses()
    responses[406] = {"model": ApiErrorResponse}
    responses[200] = {
        "description": "Ordered JSON page or Server-Sent Events stream.",
        "content": {
            "text/event-stream": {
                "schema": {"type": "string"},
                "example": (
                    "id: 3\n"
                    "event: NODE_STARTED\n"
                    'data: {"run_id":"run-1","sequence":3}\n\n'
                ),
            }
        },
    }
    return responses


def _json_error(mapping: HttpErrorMapping) -> JSONResponse:
    return JSONResponse(
        status_code=mapping.status_code,
        content=mapping.response.model_dump(mode="json"),
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _fixed_error(
    *,
    status_code: int,
    code: Literal[
        "CONFLICT",
        "FORBIDDEN",
        "IDEMPOTENCY_CONFLICT",
        "INTERNAL_ERROR",
        "INVALID_ARGUMENT",
        "INVALID_REQUEST",
        "NOT_FOUND",
        "SERVICE_UNAVAILABLE",
    ],
    message: str,
    retryable: bool = False,
) -> HttpErrorMapping:
    return HttpErrorMapping(
        status_code=status_code,
        response=ApiErrorResponse(
            error=ApiErrorDetail(
                code=code,
                message=message,
                retryable=retryable,
            )
        ),
    )


def _map_artifact_error(exc: ArtifactRegistryError) -> HttpErrorMapping:
    if exc.code == "NOT_FOUND":
        return map_exception_to_http_error(
            RunServiceError("NOT_FOUND", "artifact was not found")
        )
    if exc.code in {"PERMISSION_DENIED", "FORBIDDEN"}:
        return map_exception_to_http_error(
            RunServiceError("FORBIDDEN", "artifact access was denied")
        )
    if exc.retryable:
        return map_exception_to_http_error(
            RunServiceError(
                "ARTIFACT_DEPENDENCY_UNAVAILABLE",
                "artifact dependency is unavailable",
                retryable=True,
            )
        )
    if exc.code in {
        "ARTIFACT_NOT_AVAILABLE",
        "CASE_MISMATCH",
        "INVALID_ARGUMENT",
        "INVALID_STATE",
    }:
        return map_exception_to_http_error(
            RunServiceError("INVALID_REQUEST", "artifact input is invalid")
        )
    return map_exception_to_http_error(RuntimeError("artifact gateway failed"))


def _map_transport_exception(exc: Exception) -> HttpErrorMapping:
    if isinstance(exc, SseNegotiationError):
        return _fixed_error(
            status_code=406,
            code="INVALID_ARGUMENT",
            message="requested response representation is not acceptable",
        )
    if isinstance(exc, ArtifactRegistryError):
        return _map_artifact_error(exc)
    if isinstance(
        exc,
        (
            RunServiceError,
            PublicValueValidationError,
            ApiRequestValidationError,
        ),
    ):
        return map_exception_to_http_error(exc)
    return map_exception_to_http_error(RuntimeError("internal"))


class _SanitizedExceptionMiddleware:
    """Return a fixed error without re-raising private exception text."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracked_send)
        except Exception as exc:
            if response_started:
                raise
            response = _json_error(_map_transport_exception(exc))
            await response(scope, receive, send)


def _authorization_headers(scope: Any) -> list[bytes]:
    return [
        value
        for name, value in scope.get("headers", ())
        if isinstance(name, bytes)
        and isinstance(value, bytes)
        and name.lower() == b"authorization"
    ]


def _extract_bearer_token(scope: Any) -> str:
    values = _authorization_headers(scope)
    if len(values) != 1 or len(values[0]) > _MAX_BEARER_TOKEN_BYTES + 7:
        raise RunServiceError("FORBIDDEN", "authentication failed")
    try:
        header = values[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise RunServiceError("FORBIDDEN", "authentication failed") from exc
    if not header.lower().startswith("bearer "):
        raise RunServiceError("FORBIDDEN", "authentication failed")
    token = header[7:]
    if (
        not token
        or len(token.encode("ascii")) > _MAX_BEARER_TOKEN_BYTES
        or not _BEARER_TOKEN.fullmatch(token)
    ):
        raise RunServiceError("FORBIDDEN", "authentication failed")
    return token


class _AuthenticationMiddleware:
    """Authenticate protected paths before body buffering or JSON parsing."""

    def __init__(self, app: Any, *, authenticator: PrincipalAuthenticator) -> None:
        self.app = app
        self.authenticator = authenticator

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and str(scope.get("path", "")).startswith(
            "/api/"
        ):
            token = _extract_bearer_token(scope)
            principal = await run_in_threadpool(
                self.authenticator.authenticate,
                token,
            )
            if not isinstance(principal, AuthenticatedPrincipal):
                raise RunServiceError(
                    "AUTHENTICATION_CONFIGURATION_ERROR",
                    "authenticator returned an invalid principal",
                )
            scope["puncture.authenticated_principal"] = principal
        await self.app(scope, receive, send)


def _validate_authorized_case(
    case: AuthorizedCase,
    *,
    principal: AuthenticatedPrincipal,
    case_id: str,
    project_id: str | None = None,
) -> None:
    if (
        not isinstance(case, AuthorizedCase)
        or case.tenant_id != principal.tenant_id
        or case.case_id != case_id
    ):
        raise RunServiceError(
            "AUTHORIZATION_CONFIGURATION_ERROR",
            "authorizer returned an inconsistent case",
        )
    if project_id is not None and case.project_id != project_id:
        raise RunServiceError("FORBIDDEN", "run project access was denied")


def _validate_artifact_views(
    views: Sequence[ArtifactPublicView],
    *,
    artifact_ids: tuple[str, ...],
    case_id: str,
) -> tuple[ArtifactPublicView, ...]:
    normalized = tuple(views)
    if (
        len(normalized) != len(artifact_ids)
        or any(not isinstance(item, ArtifactPublicView) for item in normalized)
        or tuple(item.artifact_id for item in normalized) != artifact_ids
    ):
        raise RunServiceError(
            "ARTIFACT_GATEWAY_CONTRACT_ERROR",
            "artifact gateway returned inconsistent metadata",
        )
    if any(
        item.case_id != case_id or item.status is not ArtifactStatus.AVAILABLE
        for item in normalized
    ):
        raise RunServiceError("INVALID_REQUEST", "artifact input is invalid")
    return normalized


def create_app(
    run_service: RunService,
    *,
    authenticator: PrincipalAuthenticator,
    authorizer: ResourceAuthorizer,
    artifact_gateway: ArtifactAccessGateway | None = None,
    health_probe: HealthProbe | None = None,
    metrics: HttpMetrics | None = None,
    sse_config: SseConfig | None = None,
    sse_metrics: SseMetrics | None = None,
    max_request_body_bytes: int = 1024 * 1024,
    allow_test_controls: bool = False,
    startup_hooks: Sequence[Callable[[], None]] = (),
) -> FastAPI:
    """Create one isolated application with fail-closed injected authority."""

    if run_service is None:
        raise TypeError("run_service is required")
    if authenticator is None:
        raise TypeError("authenticator is required")
    if authorizer is None:
        raise TypeError("authorizer is required")
    if sse_config is not None and not isinstance(sse_config, SseConfig):
        raise TypeError("sse_config must be an SseConfig")
    if sse_metrics is not None and not isinstance(sse_metrics, SseMetrics):
        raise TypeError("sse_metrics must be an SseMetrics")
    if not isinstance(allow_test_controls, bool):
        raise TypeError("allow_test_controls must be a boolean")
    hooks = tuple(startup_hooks)
    if any(not callable(hook) for hook in hooks):
        raise TypeError("startup_hooks must contain callables")
    dependencies = _Dependencies(
        run_service=run_service,
        authenticator=authenticator,
        authorizer=authorizer,
        artifact_gateway=(
            artifact_gateway
            if artifact_gateway is not None
            else _UnavailableArtifactGateway()
        ),
        health_probe=(
            health_probe
            if health_probe is not None
            else (
                AlwaysUpHealthProbe()
                if artifact_gateway is not None
                else _DegradedHealthProbe()
            )
        ),
        allow_test_controls=allow_test_controls,
    )
    metric_store = metrics if metrics is not None else HttpMetrics()
    event_stream_config = sse_config if sse_config is not None else SseConfig()
    event_stream_metrics = sse_metrics if sse_metrics is not None else SseMetrics()
    event_stream_limiter = SseConnectionLimiter(
        max_connections=event_stream_config.max_connections,
        max_per_tenant=event_stream_config.max_connections_per_tenant,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        for hook in hooks:
            await run_in_threadpool(hook)
        yield

    app = FastAPI(
        title="Puncture R&D Agent API",
        version="0.5.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.puncture_dependencies = dependencies
    app.state.puncture_metrics = metric_store
    app.state.puncture_sse_metrics = event_stream_metrics
    app.state.puncture_sse_limiter = event_stream_limiter
    app.add_middleware(
        RawBodyAdmissionMiddleware,
        max_body_bytes=max_request_body_bytes,
    )
    app.add_middleware(
        _AuthenticationMiddleware,
        authenticator=dependencies.authenticator,
    )
    app.add_middleware(_SanitizedExceptionMiddleware)
    app.add_middleware(HttpMetricsMiddleware, metrics=metric_store)

    bearer = HTTPBearer(auto_error=False, scheme_name="BearerAuth")

    def current_principal(
        request: Request,
        _: HTTPAuthorizationCredentials | None = Security(bearer),
    ) -> AuthenticatedPrincipal:
        principal = request.scope.get("puncture.authenticated_principal")
        if not isinstance(principal, AuthenticatedPrincipal):
            raise RunServiceError(
                "AUTHENTICATION_CONFIGURATION_ERROR",
                "authenticator returned an invalid principal",
            )
        return principal

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        locations = {
            str(error.get("loc", ("body",))[0])
            for error in exc.errors()
            if error.get("loc")
        }
        if locations and locations <= {"path", "query", "header"}:
            return _json_error(
                _fixed_error(
                    status_code=400,
                    code="INVALID_ARGUMENT",
                    message="request argument is invalid",
                )
            )
        return _json_error(
            _fixed_error(
                status_code=422,
                code="INVALID_REQUEST",
                message="request validation failed",
            )
        )

    @app.exception_handler(StarletteHttpException)
    async def http_exception_handler(
        _: Request,
        exc: StarletteHttpException,
    ) -> JSONResponse:
        if exc.status_code == 404:
            mapping = _fixed_error(
                status_code=404,
                code="NOT_FOUND",
                message="resource was not found",
            )
        elif exc.status_code == 405:
            mapping = _fixed_error(
                status_code=405,
                code="INVALID_ARGUMENT",
                message="request method is not allowed",
            )
        elif exc.status_code == 415:
            mapping = _fixed_error(
                status_code=415,
                code="INVALID_REQUEST",
                message="request content type is not supported",
            )
        elif exc.status_code in {401, 403}:
            mapping = _fixed_error(
                status_code=403,
                code="FORBIDDEN",
                message="request is not authorized",
            )
        else:
            mapping = map_exception_to_http_error(RuntimeError("http error"))
        return _json_error(mapping)

    def authorized_run(
        run_id: str,
        principal: AuthenticatedPrincipal,
        permission: ApiPermission,
    ) -> RunSnapshot:
        snapshot = dependencies.run_service.get_run(
            run_id,
            tenant_id=principal.tenant_id,
        )
        case = dependencies.authorizer.require_run(
            principal,
            snapshot=snapshot,
            permission=permission,
        )
        _validate_authorized_case(
            case,
            principal=principal,
            case_id=snapshot.request.case_id,
            project_id=(
                snapshot.request.metadata.get("project_id")
                if isinstance(snapshot.request.metadata.get("project_id"), str)
                else None
            ),
        )
        if not isinstance(snapshot.request.metadata.get("project_id"), str):
            raise RunServiceError("FORBIDDEN", "run project binding is unavailable")
        return snapshot

    @app.post(
        "/api/v1/runs",
        operation_id="createRun",
        response_model=RunSnapshotResponse,
        responses=_error_responses(),
        tags=["runs"],
    )
    def create_run(
        body: RunCreateBody,
        principal: AuthenticatedPrincipal = Depends(current_principal),
    ) -> RunSnapshotResponse:
        request = body.to_runtime_request(
            principal,
            allow_test_controls=dependencies.allow_test_controls,
        )
        case = dependencies.authorizer.require_case(
            principal,
            case_id=request.case_id,
            permission=ApiPermission.RUN_CREATE,
        )
        _validate_authorized_case(
            case,
            principal=principal,
            case_id=request.case_id,
        )
        if request.artifact_ids:
            views = dependencies.artifact_gateway.require_artifacts(
                principal,
                case=case,
                artifact_ids=request.artifact_ids,
                permission=ApiPermission.ARTIFACT_USE,
            )
            _validate_artifact_views(
                views,
                artifact_ids=request.artifact_ids,
                case_id=request.case_id,
            )
        request = replace(
            request,
            metadata={**request.metadata, "project_id": case.project_id},
        )
        return RunSnapshotResponse.from_runtime(
            dependencies.run_service.create_run(request)
        )

    @app.get(
        "/api/v1/runs/{run_id}",
        operation_id="getRun",
        response_model=RunSnapshotResponse,
        responses=_error_responses(),
        tags=["runs"],
    )
    def get_run(
        run_id: str = Path(
            min_length=1,
            max_length=255,
            pattern=_RESOURCE_ID_PATTERN,
        ),
        principal: AuthenticatedPrincipal = Depends(current_principal),
    ) -> RunSnapshotResponse:
        return RunSnapshotResponse.from_runtime(
            authorized_run(run_id, principal, ApiPermission.RUN_READ)
        )

    @app.get(
        "/api/v1/runs/{run_id}/events",
        operation_id="listRunEvents",
        response_model=list[RunEventResponse],
        responses=_event_responses(),
        description=(
            "Returns a bounded JSON page by default. Send "
            "Accept: text/event-stream for SSE; Last-Event-ID and "
            "after_sequence are exclusive cursors and must agree when both "
            "are present."
        ),
        tags=["runs"],
    )
    async def list_run_events(
        request: Request,
        response: Response,
        run_id: str = Path(
            min_length=1,
            max_length=255,
            pattern=_RESOURCE_ID_PATTERN,
        ),
        after_sequence: int | None = Query(
            default=None,
            ge=0,
            le=2**63 - 1,
        ),
        limit: int = Query(default=512, ge=1, le=512),
        principal: AuthenticatedPrincipal = Depends(current_principal),
    ) -> Any:
        representation = negotiate_event_representation(request.scope)
        cursor = resolve_event_cursor(
            request.scope,
            after_sequence=after_sequence,
            representation=representation,
        )
        await run_in_threadpool(
            authorized_run,
            run_id,
            principal,
            ApiPermission.RUN_EVENTS_READ,
        )
        page_limit = limit if representation == "json" else event_stream_config.page_size
        page = await run_in_threadpool(
            dependencies.run_service.get_event_page,
            run_id,
            tenant_id=principal.tenant_id,
            after_sequence=cursor.cursor,
            limit=page_limit,
        )
        frames = encode_event_page(
            page,
            run_id=run_id,
            after_sequence=cursor.cursor,
        )
        if representation == "json":
            response.headers["Vary"] = "Accept"
            return [
                RunEventResponse.from_runtime(event)
                for event, _ in frames
            ]

        stream_bearer_token = _extract_bearer_token(request.scope)
        lease = event_stream_limiter.try_acquire(principal.tenant_id)

        def authorize_stream() -> RunSnapshot:
            refreshed_principal = dependencies.authenticator.authenticate(
                stream_bearer_token
            )
            if not isinstance(refreshed_principal, AuthenticatedPrincipal):
                raise RunServiceError(
                    "AUTHENTICATION_CONFIGURATION_ERROR",
                    "authenticator returned an invalid principal",
                )
            if (
                refreshed_principal.tenant_id != principal.tenant_id
                or refreshed_principal.principal_id != principal.principal_id
            ):
                raise RunServiceError("FORBIDDEN", "stream identity changed")
            return authorized_run(
                run_id,
                refreshed_principal,
                ApiPermission.RUN_EVENTS_READ,
            )

        def read_stream_page(
            page_cursor: int,
            page_size: int,
        ) -> RunEventPage:
            return dependencies.run_service.get_event_page(
                run_id,
                tenant_id=principal.tenant_id,
                after_sequence=page_cursor,
                limit=page_size,
            )

        try:
            stream = stream_event_pages(
                run_id=run_id,
                initial_page=page,
                initial_frames=frames,
                after_sequence=cursor.cursor,
                authorize=authorize_stream,
                read_page=read_stream_page,
                config=event_stream_config,
                lease=lease,
                metrics=event_stream_metrics,
                cursor_source=cursor.source,
            )
            return SseStreamingResponse(
                stream,
                lease=lease,
                headers={
                    "Cache-Control": (
                        "no-cache, no-store, must-revalidate, no-transform"
                    ),
                    "X-Accel-Buffering": "no",
                    "X-Content-Type-Options": "nosniff",
                    "Vary": "Accept",
                },
            )
        except BaseException:
            lease.release()
            raise

    @app.post(
        "/api/v1/runs/{run_id}/approvals/{approval_id}",
        operation_id="decideRunApproval",
        response_model=RunSnapshotResponse,
        responses=_error_responses(),
        tags=["runs"],
    )
    def decide_run_approval(
        body: ApprovalBody,
        run_id: str = Path(
            min_length=1,
            max_length=255,
            pattern=_RESOURCE_ID_PATTERN,
        ),
        approval_id: str = Path(
            min_length=1,
            max_length=128,
            pattern=_RESOURCE_ID_PATTERN,
        ),
        principal: AuthenticatedPrincipal = Depends(current_principal),
    ) -> RunSnapshotResponse:
        authorized_run(run_id, principal, ApiPermission.RUN_APPROVE)
        decision = body.to_runtime_decision(
            approval_id=approval_id,
            principal=principal,
        )
        snapshot = dependencies.run_service.approve(
            run_id,
            decision,
            tenant_id=principal.tenant_id,
        )
        return RunSnapshotResponse.from_runtime(snapshot)

    @app.post(
        "/api/v1/runs/{run_id}/cancel",
        operation_id="cancelRun",
        response_model=RunSnapshotResponse,
        responses=_error_responses(),
        tags=["runs"],
    )
    def cancel_run(
        run_id: str = Path(
            min_length=1,
            max_length=255,
            pattern=_RESOURCE_ID_PATTERN,
        ),
        principal: AuthenticatedPrincipal = Depends(current_principal),
    ) -> RunSnapshotResponse:
        authorized_run(run_id, principal, ApiPermission.RUN_CANCEL)
        return RunSnapshotResponse.from_runtime(
            dependencies.run_service.cancel(
                run_id,
                tenant_id=principal.tenant_id,
            )
        )

    @app.post(
        "/api/v1/runs/{run_id}/resume",
        operation_id="resumeRun",
        response_model=RunSnapshotResponse,
        responses=_error_responses(),
        tags=["runs"],
    )
    def resume_run(
        run_id: str = Path(
            min_length=1,
            max_length=255,
            pattern=_RESOURCE_ID_PATTERN,
        ),
        principal: AuthenticatedPrincipal = Depends(current_principal),
    ) -> RunSnapshotResponse:
        authorized_run(run_id, principal, ApiPermission.RUN_RESUME)
        return RunSnapshotResponse.from_runtime(
            dependencies.run_service.resume(
                run_id,
                tenant_id=principal.tenant_id,
            )
        )

    @app.get(
        "/api/v1/artifacts/{artifact_id}/metadata",
        operation_id="getArtifactMetadata",
        response_model=ArtifactMetadataResponse,
        responses=_error_responses(),
        tags=["artifacts"],
    )
    def get_artifact_metadata(
        artifact_id: str = Path(
            min_length=1,
            max_length=255,
            pattern=_ARTIFACT_RESOURCE_ID_PATTERN,
        ),
        principal: AuthenticatedPrincipal = Depends(current_principal),
    ) -> ArtifactMetadataResponse:
        artifact = dependencies.artifact_gateway.get_metadata(
            principal,
            artifact_id=artifact_id,
            permission=ApiPermission.ARTIFACT_READ,
        )
        if (
            not isinstance(artifact, ArtifactPublicView)
            or artifact.artifact_id != artifact_id
        ):
            raise RunServiceError(
                "ARTIFACT_GATEWAY_CONTRACT_ERROR",
                "artifact gateway returned inconsistent metadata",
            )
        return ArtifactMetadataResponse.from_contract(artifact)

    @app.get(
        "/health",
        operation_id="getHealth",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        tags=["operations"],
    )
    def get_health() -> HealthResponse | JSONResponse:
        try:
            status = dependencies.health_probe.status()
            if status not in {"UP", "DEGRADED"}:
                raise ValueError("health probe returned an invalid status")
            return HealthResponse(status=status)
        except Exception:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(status="DOWN").model_dump(mode="json"),
                headers={"Cache-Control": "no-store"},
            )

    @app.get(
        "/metrics",
        operation_id="getMetrics",
        response_class=PlainTextResponse,
        tags=["operations"],
    )
    def get_metrics() -> PlainTextResponse:
        return PlainTextResponse(
            metric_store.render() + event_stream_metrics.render(),
            media_type="text/plain; version=0.0.4",
            headers={"Cache-Control": "no-store"},
        )

    return app


__all__ = [
    "AlwaysUpHealthProbe",
    "ApiPermission",
    "ArtifactAccessGateway",
    "AuthorizedCase",
    "HealthProbe",
    "PrincipalAuthenticator",
    "ResourceAuthorizer",
    "RunService",
    "create_app",
]
