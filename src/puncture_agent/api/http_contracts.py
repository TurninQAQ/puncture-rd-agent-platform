"""Pydantic v2 adapters for the fixed framework-neutral Run contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from contracts.artifacts import ArtifactPublicView
from contracts.enums import ArtifactStatus, ArtifactType
from puncture_agent.runtime.models import (
    ApprovalDecision,
    EventType,
    RunEvent,
    RunRequest,
    RunSnapshot,
    RunStatus,
)
from puncture_agent.runtime.service import RunServiceError

from .privacy import (
    PublicValueValidationError,
    REDACTED,
    normalize_public_key,
    redact_public_value,
    validate_public_json_input,
)


_CASE_ID = Annotated[
    StrictStr,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$",
    ),
]
_QUERY = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=16_384),
]
_IDEMPOTENCY_KEY = Annotated[
    StrictStr,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$",
    ),
]
_ARTIFACT_ID = Annotated[
    StrictStr,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
_PUBLIC_METADATA_TEXT = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
_TEST_CONTROL_KEYS = {
    "approval_id",
    "dependency_timeout",
    "force_failure",
    "requires_approval",
}
_AUTHORITY_METADATA_KEYS = {
    "access_scope",
    "access_scopes",
    "allowed_artifacts",
    "allowed_cases",
    "case_id",
    "case_ids",
    "permissions",
    "principal_id",
    "project_id",
    "project_ids",
    "role",
    "roles",
    "scope",
    "scopes",
    "tenant_id",
}
_PUBLIC_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
PublicApiErrorCode = Literal[
    "CONFLICT",
    "FORBIDDEN",
    "IDEMPOTENCY_CONFLICT",
    "INTERNAL_ERROR",
    "INVALID_ARGUMENT",
    "INVALID_REQUEST",
    "NOT_FOUND",
    "SERVICE_UNAVAILABLE",
]


class _FrozenApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiRequestValidationError(ValueError):
    """A validated HTTP body cannot be converted into a trusted runtime request."""


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    tenant_id: str
    principal_id: str
    access_scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("tenant_id is required")
        if not isinstance(self.principal_id, str) or not self.principal_id.strip():
            raise ValueError("principal_id is required")
        tenant_id = self.tenant_id.strip()
        principal_id = self.principal_id.strip()
        if not _PUBLIC_IDENTIFIER.fullmatch(tenant_id):
            raise ValueError("tenant_id contains unsupported characters")
        if not _PUBLIC_IDENTIFIER.fullmatch(principal_id):
            raise ValueError("principal_id contains unsupported characters")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "principal_id", principal_id)
        if isinstance(self.access_scopes, (str, bytes, bytearray)):
            raise ValueError("access_scopes must be a sequence of scope strings")
        try:
            scopes = tuple(self.access_scopes)
        except TypeError as exc:
            raise ValueError(
                "access_scopes must be a sequence of scope strings"
            ) from exc
        if any(
            not isinstance(scope, str)
            or not scope.strip()
            or len(scope.strip()) > 128
            or not _PUBLIC_IDENTIFIER.fullmatch(scope.strip())
            for scope in scopes
        ):
            raise ValueError("access scopes must be non-empty strings up to 128 characters")
        normalized_scopes = tuple(scope.strip() for scope in scopes)
        if len(set(normalized_scopes)) != len(normalized_scopes):
            raise ValueError("access scopes must be unique")
        object.__setattr__(self, "access_scopes", normalized_scopes)


class RunCreateBody(_FrozenApiModel):
    case_id: _CASE_ID
    user_query: _QUERY
    task_type: Literal["DATA_MODEL_VALIDATION", "PLANNING_SAFETY"]
    idempotency_key: _IDEMPOTENCY_KEY
    artifact_ids: tuple[_ARTIFACT_ID, ...] = Field(default=(), max_length=128)
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)

    @field_validator("user_query")
    @classmethod
    def _query_has_no_embedded_secret(cls, value: str) -> str:
        if redact_public_value(value) == REDACTED:
            raise ValueError("user_query contains a forbidden credential or location")
        return value

    @field_validator("artifact_ids")
    @classmethod
    def _artifact_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("artifact_ids must be unique")
        return value

    @model_validator(mode="after")
    def _metadata_is_bounded_json(self) -> "RunCreateBody":
        validate_public_json_input(
            self.metadata,
            forbidden_keys=_AUTHORITY_METADATA_KEYS,
        )
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 1024 * 1024:
            raise ValueError("normalized request body exceeds 1 MiB")
        return self

    def to_runtime_request(
        self,
        principal: AuthenticatedPrincipal,
        *,
        allow_test_controls: bool = False,
    ) -> RunRequest:
        controls = sorted(
            key
            for key in self.metadata
            if normalize_public_key(key) in _TEST_CONTROL_KEYS
        )
        if controls and not allow_test_controls:
            raise ApiRequestValidationError(
                "test-only request metadata controls are disabled"
            )
        authority_keys = sorted(
            key
            for key in self.metadata
            if normalize_public_key(key) in _AUTHORITY_METADATA_KEYS
        )
        if authority_keys:
            raise ApiRequestValidationError(
                "request metadata cannot override authenticated authority"
            )
        metadata = dict(self.metadata)
        if principal.access_scopes:
            metadata["access_scopes"] = list(principal.access_scopes)
        return RunRequest(
            case_id=self.case_id,
            user_query=self.user_query,
            task_type=self.task_type,
            idempotency_key=self.idempotency_key,
            tenant_id=principal.tenant_id,
            principal_id=principal.principal_id,
            artifact_ids=self.artifact_ids,
            metadata=metadata,
        )


class ApprovalBody(_FrozenApiModel):
    approved: StrictBool
    comment: Annotated[StrictStr, StringConstraints(max_length=2048)] = ""

    @field_validator("comment")
    @classmethod
    def _comment_has_no_embedded_secret(cls, value: str) -> str:
        if redact_public_value(value) == REDACTED:
            raise ValueError("comment contains a forbidden credential or location")
        return value

    def to_runtime_decision(
        self,
        *,
        approval_id: str,
        principal: AuthenticatedPrincipal,
    ) -> ApprovalDecision:
        if not isinstance(approval_id, str) or not _PUBLIC_IDENTIFIER.fullmatch(
            approval_id.strip()
        ):
            raise ApiRequestValidationError("approval_id is invalid")
        return ApprovalDecision(
            approval_id=approval_id.strip(),
            approved=self.approved,
            principal_id=principal.principal_id,
            comment=self.comment,
        )


class ApiErrorDetail(_FrozenApiModel):
    code: PublicApiErrorCode
    message: Annotated[StrictStr, StringConstraints(min_length=1, max_length=1024)]
    retryable: StrictBool = False
    field_path: StrictStr | None = None
    dependency: StrictStr | None = None
    details: dict[StrictStr, StrictStr] = Field(default_factory=dict)


class ApiErrorResponse(_FrozenApiModel):
    error: ApiErrorDetail


@dataclass(frozen=True, slots=True)
class HttpErrorMapping:
    status_code: int
    response: ApiErrorResponse


class RunRequestView(_FrozenApiModel):
    case_id: StrictStr
    user_query: StrictStr
    task_type: StrictStr
    idempotency_key: StrictStr
    tenant_id: StrictStr
    principal_id: StrictStr
    artifact_ids: tuple[StrictStr, ...]
    metadata: dict[str, Any]


class RunSnapshotResponse(_FrozenApiModel):
    run_id: StrictStr
    request: RunRequestView
    status: RunStatus
    trace_id: StrictStr
    created_at: StrictStr
    updated_at: StrictStr
    final_report: dict[str, Any]
    checkpoint: dict[str, Any]
    approval_id: StrictStr | None
    error: dict[str, Any] | None

    @classmethod
    def from_runtime(cls, snapshot: RunSnapshot) -> "RunSnapshotResponse":
        request = snapshot.request
        public_metadata = redact_public_value(request.metadata)
        for key in tuple(public_metadata):
            if normalize_public_key(key) in _AUTHORITY_METADATA_KEYS:
                public_metadata[key] = REDACTED
        error = (
            redact_public_value(snapshot.error)
            if snapshot.error is not None
            else None
        )
        if error is not None and "message" in error:
            error["message"] = REDACTED
        return cls(
            run_id=snapshot.run_id,
            request=RunRequestView(
                case_id=request.case_id,
                user_query="[REDACTED]",
                task_type=request.task_type,
                idempotency_key="[REDACTED]",
                tenant_id=request.tenant_id,
                principal_id=request.principal_id,
                artifact_ids=request.artifact_ids,
                metadata=public_metadata,
            ),
            status=snapshot.status,
            trace_id=snapshot.trace_id,
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
            final_report=redact_public_value(snapshot.final_report),
            checkpoint=redact_public_value(snapshot.checkpoint),
            approval_id=(
                redact_public_value(snapshot.approval_id)
                if snapshot.approval_id is not None
                else None
            ),
            error=error,
        )


class RunEventResponse(_FrozenApiModel):
    run_id: StrictStr
    sequence: int = Field(ge=1)
    event_type: EventType
    node_name: StrictStr | None
    timestamp: StrictStr
    payload: dict[str, Any]
    trace_id: StrictStr

    @classmethod
    def from_runtime(cls, event: RunEvent) -> "RunEventResponse":
        node_name = (
            redact_public_value(event.node_name)
            if event.node_name is not None
            else None
        )
        return cls(
            run_id=event.run_id,
            sequence=event.sequence,
            event_type=event.event_type,
            node_name=node_name,
            timestamp=event.timestamp,
            payload=redact_public_value(event.payload),
            trace_id=event.trace_id,
        )


class ArtifactMetadataResponse(_FrozenApiModel):
    artifact_id: _ARTIFACT_ID
    case_id: _CASE_ID
    artifact_type: ArtifactType
    status: ArtifactStatus
    producer_name: _PUBLIC_METADATA_TEXT
    producer_version: _PUBLIC_METADATA_TEXT
    geometry_fingerprint: _PUBLIC_METADATA_TEXT | None

    @field_validator(
        "producer_name",
        "producer_version",
        "geometry_fingerprint",
    )
    @classmethod
    def _metadata_text_is_public(cls, value: str | None) -> str | None:
        if value is not None:
            validate_public_json_input(value, max_depth=1, max_nodes=1)
        return value

    @classmethod
    def from_contract(
        cls,
        artifact: ArtifactPublicView,
    ) -> "ArtifactMetadataResponse":
        if not isinstance(artifact, ArtifactPublicView):
            raise TypeError("artifact metadata must be an ArtifactPublicView")
        return cls(
            artifact_id=artifact.artifact_id,
            case_id=artifact.case_id,
            artifact_type=artifact.artifact_type,
            status=artifact.status,
            producer_name=artifact.producer_name,
            producer_version=artifact.producer_version,
            geometry_fingerprint=artifact.geometry_fingerprint,
        )


class HealthResponse(_FrozenApiModel):
    status: Literal["UP", "DOWN", "DEGRADED"]


def map_exception_to_http_error(exc: BaseException) -> HttpErrorMapping:
    """Map internal exceptions to stable, sanitized public HTTP errors."""

    if isinstance(exc, RunServiceError):
        status_by_code = {
            "CONFLICT": 409,
            "FORBIDDEN": 403,
            "IDEMPOTENCY_CONFLICT": 409,
            "INVALID_ARGUMENT": 400,
            "INVALID_REQUEST": 422,
            "NOT_FOUND": 404,
        }
        public_message_by_code = {
            "CONFLICT": "request conflicts with the current run state",
            "FORBIDDEN": "request is not authorized",
            "IDEMPOTENCY_CONFLICT": "idempotency key conflicts with an existing request",
            "INVALID_ARGUMENT": "request argument is invalid",
            "INVALID_REQUEST": "request validation failed",
            "NOT_FOUND": "resource was not found",
        }
        if exc.code in status_by_code:
            status_code = status_by_code[exc.code]
            code = exc.code
            message = public_message_by_code[exc.code]
        elif exc.retryable:
            status_code = 503
            code = "SERVICE_UNAVAILABLE"
            message = "service dependency is unavailable"
        else:
            status_code = 500
            code = "INTERNAL_ERROR"
            message = "internal service error"
        retryable = exc.retryable
    elif isinstance(
        exc,
        (ValidationError, PublicValueValidationError, ApiRequestValidationError),
    ):
        status_code = 422
        code = "INVALID_REQUEST"
        message = "request validation failed"
        retryable = False
    else:
        status_code = 500
        code = "INTERNAL_ERROR"
        message = "internal service error"
        retryable = False
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


__all__ = [
    "ApiErrorDetail",
    "ApiErrorResponse",
    "ApiRequestValidationError",
    "ApprovalBody",
    "ArtifactMetadataResponse",
    "AuthenticatedPrincipal",
    "HealthResponse",
    "HttpErrorMapping",
    "PublicApiErrorCode",
    "RunCreateBody",
    "RunEventResponse",
    "RunRequestView",
    "RunSnapshotResponse",
    "map_exception_to_http_error",
]
