"""MCP-facing runtime for the strongly typed internal tool registry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import time
from typing import Any, Callable, Mapping

from contracts.common import ToolResponseEnvelope
from contracts.enums import ErrorCode, ToolExecutionStatus
from contracts.errors import ErrorDetail
from puncture_agent.tooling.registry import ToolDefinition, ToolRegistry

from .codec import (
    ArtifactResolver,
    ContractDecodeError,
    decode_tool_context,
    decode_tool_request,
    to_mcp_safe_primitive,
)
from .ledger import (
    ToolReplayBusy,
    ToolReplayConflict,
    ToolReplayDecision,
    ToolReplayLedger,
    ToolReplayLedgerError,
    ToolReplayUncertain,
)
from .schema import envelope_schema, request_schema


SERVER_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "case-data": (
        "inspect_case_metadata",
        "convert_mcs_to_nifti",
        "validate_label_schema",
    ),
    "segmentation": (
        "run_segmentation",
        "validate_segmentation_result",
        "extract_skin_surface",
    ),
    "planning-safety": (
        "generate_candidate_paths",
        "evaluate_path_safety",
        "evaluate_intraoperative_risk",
        "verify_skin_penetration",
    ),
}

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class McpPrincipal:
    """Transport-authenticated identity passed to the tool boundary."""

    subject: str
    allowed_case_ids: tuple[str, ...]
    allowed_tools: tuple[str, ...] = ("*",)

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValueError("principal subject is required")
        if not self.allowed_case_ids:
            raise ValueError("at least one allowed case ID is required")
        if not self.allowed_tools:
            raise ValueError("at least one allowed tool is required")

    def permits(self, *, tool_name: str, case_id: str, caller: str) -> bool:
        caller_matches = caller == self.subject
        case_allowed = "*" in self.allowed_case_ids or case_id in self.allowed_case_ids
        tool_allowed = "*" in self.allowed_tools or tool_name in self.allowed_tools
        return caller_matches and case_allowed and tool_allowed


@dataclass(frozen=True, slots=True)
class McpCallResult:
    content: tuple[dict[str, Any], ...]
    structured_content: dict[str, Any]
    is_error: bool
    meta: dict[str, Any]

    def to_protocol_result(self) -> dict[str, Any]:
        return {
            "content": [dict(item) for item in self.content],
            "structuredContent": dict(self.structured_content),
            "isError": self.is_error,
            "_meta": dict(self.meta),
        }


class McpToolRuntime:
    """Expose one logical MCP server over an internal :class:`ToolRegistry`."""

    def __init__(
        self,
        registry: ToolRegistry,
        artifact_resolver: ArtifactResolver,
        *,
        server_name: str,
        replay_ledger: ToolReplayLedger | None = None,
        replay_authorizer: Callable[[str, Any, McpPrincipal], bool] | None = None,
        replay_response_validator: Callable[
            [str, Any, Mapping[str, Any], McpPrincipal], bool
        ]
        | None = None,
    ) -> None:
        if server_name not in SERVER_TOOL_NAMES:
            raise ValueError(f"unknown MCP server: {server_name}")
        self._registry = registry
        self._artifact_resolver = artifact_resolver
        if replay_ledger is not None and not isinstance(
            replay_ledger,
            ToolReplayLedger,
        ):
            raise TypeError("replay_ledger must implement ToolReplayLedger")
        if replay_ledger is not None and replay_authorizer is None:
            raise ValueError(
                "replay_authorizer is required with a durable replay ledger"
            )
        if replay_ledger is not None and replay_response_validator is None:
            raise ValueError(
                "replay_response_validator is required with a durable replay ledger"
            )
        self._replay_ledger = replay_ledger
        self._replay_authorizer = replay_authorizer
        self._replay_response_validator = replay_response_validator
        self.server_name = server_name
        self._tool_names = SERVER_TOOL_NAMES[server_name]
        registered = {definition.name for definition in registry.list_definitions()}
        missing = sorted(set(self._tool_names) - registered)
        if missing:
            raise ValueError("registry is missing MCP tools: " + ", ".join(missing))
        if replay_ledger is not None:
            max_timeout_seconds = max(
                registry.get_definition(name).default_timeout_ms
                for name in self._tool_names
            ) / 1000.0
            if replay_ledger.claim_ttl_seconds <= max_timeout_seconds:
                raise ValueError(
                    "replay ledger claim TTL must exceed every server tool timeout"
                )

    @property
    def tool_names(self) -> tuple[str, ...]:
        return self._tool_names

    def list_tools(self) -> tuple[dict[str, Any], ...]:
        tools = []
        for name in self._tool_names:
            definition = self._registry.get_definition(name)
            tools.append(
                {
                    "name": name,
                    "title": name.replace("_", " ").title(),
                    "description": definition.description,
                    "inputSchema": request_schema(definition.request_type),
                    "outputSchema": envelope_schema(definition.result_type),
                    "annotations": {
                        "readOnlyHint": definition.read_only,
                        "destructiveHint": definition.destructive,
                        "idempotentHint": definition.idempotent,
                        "openWorldHint": definition.open_world,
                    },
                    "execution": {"taskSupport": "forbidden"},
                    "_meta": {
                        "com.turninqaq/toolVersion": definition.version,
                        "com.turninqaq/server": self.server_name,
                        "com.turninqaq/defaultTimeoutMs": definition.default_timeout_ms,
                    },
                }
            )
        return tuple(tools)

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        principal: McpPrincipal,
    ) -> McpCallResult:
        if name not in self._tool_names:
            raise KeyError(f"unknown tool for {self.server_name}: {name}")
        definition = self._registry.get_definition(name)
        if not isinstance(arguments, Mapping):
            response = self._failure_from_arguments(
                definition,
                {},
                ErrorCode.INVALID_ARGUMENT,
                "tool arguments must be an object",
                field_path="$",
            )
            return self._to_call_result(response)
        try:
            context = decode_tool_context(arguments)
        except ContractDecodeError as exc:
            response = self._failure_from_arguments(
                definition,
                arguments,
                ErrorCode.INVALID_ARGUMENT,
                exc.message,
                field_path=exc.path,
            )
            return self._to_call_result(response)
        if not principal.permits(
            tool_name=name,
            case_id=context.case_id,
            caller=context.caller,
        ):
            response = self._failure(
                definition,
                request_id=context.request_id,
                trace_id=context.trace_id,
                code=ErrorCode.PERMISSION_DENIED,
                message="authenticated principal is not allowed to invoke this tool for the case",
                retryable=False,
            )
            return self._to_call_result(response)

        try:
            request = decode_tool_request(
                definition.request_type,
                arguments,
                artifact_resolver=self._artifact_resolver,
            )
        except ContractDecodeError as exc:
            response = self._failure_from_arguments(
                definition,
                arguments,
                ErrorCode.INVALID_ARGUMENT,
                exc.message,
                field_path=exc.path,
            )
            return self._to_call_result(response)
        if self._replay_authorizer is not None:
            try:
                replay_allowed = self._replay_authorizer(name, request, principal)
            except Exception:
                replay_allowed = False
            if replay_allowed is not True:
                response = self._failure(
                    definition,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    code=ErrorCode.PERMISSION_DENIED,
                    message="current tool replay authorization was denied",
                    retryable=False,
                )
                return self._to_call_result(response)

        replay_decision: ToolReplayDecision | None = None
        if self._replay_ledger is not None:
            scope_key, request_fingerprint = self._replay_identity(
                definition,
                arguments,
            )
            try:
                replay_decision = self._replay_ledger.begin(
                    scope_key,
                    request_fingerprint,
                    reclaim_expired=definition.read_only,
                )
                if replay_decision.is_replay:
                    try:
                        replay_response_allowed = self._replay_response_validator(
                            name,
                            request,
                            replay_decision.response,
                            principal,
                        )
                    except Exception:
                        replay_response_allowed = False
                    if replay_response_allowed is not True:
                        response = self._failure(
                            definition,
                            request_id=context.request_id,
                            trace_id=context.trace_id,
                            code=ErrorCode.PERMISSION_DENIED,
                            message="stored tool response is no longer authorized",
                            retryable=False,
                        )
                        return self._to_call_result(response)
                    return self._replay_call_result(
                        replay_decision,
                        definition=definition,
                        request_id=context.request_id,
                        trace_id=context.trace_id,
                    )
            except ToolReplayConflict:
                response = self._failure(
                    definition,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    code=ErrorCode.INVALID_ARGUMENT,
                    message="idempotency key was reused with a different request",
                    retryable=False,
                    field_path="context.idempotency_key",
                )
                return self._to_call_result(response)
            except ToolReplayBusy:
                response = self._failure(
                    definition,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    code=ErrorCode.DEPENDENCY_FAILED,
                    message="an identical tool execution is already in progress",
                    retryable=True,
                )
                return self._to_call_result(response)
            except ToolReplayUncertain:
                response = self._failure(
                    definition,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    code=ErrorCode.DEPENDENCY_FAILED,
                    message="tool execution requires manual replay reconciliation",
                    retryable=False,
                )
                return self._to_call_result(response)
            except ToolReplayLedgerError:
                response = self._failure(
                    definition,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    code=ErrorCode.DEPENDENCY_FAILED,
                    message="durable tool replay state is unavailable",
                    retryable=False,
                )
                return self._to_call_result(response)

        timeout_seconds = self._timeout_seconds(definition, context.deadline_epoch_ms)
        if timeout_seconds <= 0:
            self._abandon_replay(replay_decision)
            response = self._failure(
                definition,
                request_id=context.request_id,
                trace_id=context.trace_id,
                code=ErrorCode.TIMEOUT,
                message="tool deadline expired before execution",
                retryable=True,
            )
            return self._to_call_result(response)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mcp-{name}")
        future = executor.submit(self._registry.execute, name, request)
        try:
            response = future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            # A Python thread cannot be killed safely.  Do not return while a
            # write-like handler can still mutate state in the background.
            # Production native/service adapters must honor the propagated
            # deadline cooperatively so this wait remains bounded.
            executor.shutdown(wait=True, cancel_futures=True)
            if replay_decision is not None and not definition.read_only:
                self._mark_replay_uncertain(replay_decision)
                response = self._failure(
                    definition,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    code=ErrorCode.DEPENDENCY_FAILED,
                    message=(
                        "write execution exceeded its wait deadline and requires "
                        "manual replay reconciliation"
                    ),
                    retryable=False,
                )
            else:
                response = self._failure(
                    definition,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    code=ErrorCode.TIMEOUT,
                    message="tool execution exceeded its bounded deadline",
                    retryable=True,
                )
                self._abandon_replay(replay_decision)
            return self._to_call_result(response)
        except Exception as exc:  # defensive boundary around custom registries
            executor.shutdown(wait=True, cancel_futures=True)
            if replay_decision is not None and not definition.read_only:
                self._mark_replay_uncertain(replay_decision)
                response = self._failure(
                    definition,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    code=ErrorCode.DEPENDENCY_FAILED,
                    message="write execution outcome requires manual replay reconciliation",
                    retryable=False,
                )
            else:
                response = self._failure(
                    definition,
                    request_id=context.request_id,
                    trace_id=context.trace_id,
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"tool runtime rejected an unexpected {type(exc).__name__}",
                    retryable=False,
                )
                self._abandon_replay(replay_decision)
            return self._to_call_result(response)
        executor.shutdown(wait=True, cancel_futures=True)

        if response.request_id != context.request_id or response.trace_id != context.trace_id:
            response = self._failure(
                definition,
                request_id=context.request_id,
                trace_id=context.trace_id,
                code=ErrorCode.CONTRACT_VIOLATION,
                message="tool response identity does not match the request",
                retryable=False,
            )
        elif response.tool_version != definition.version:
            response = self._failure(
                definition,
                request_id=context.request_id,
                trace_id=context.trace_id,
                code=ErrorCode.CONTRACT_VIOLATION,
                message="tool response version does not match the registry definition",
                retryable=False,
            )
        call_result = self._to_call_result(response)
        if replay_decision is not None:
            if self._is_replayable_terminal(response):
                try:
                    self._replay_ledger.complete(
                        replay_decision,
                        call_result.structured_content,
                    )
                except ToolReplayLedgerError:
                    self._mark_replay_uncertain(replay_decision)
                    uncertain = self._failure(
                        definition,
                        request_id=context.request_id,
                        trace_id=context.trace_id,
                        code=ErrorCode.DEPENDENCY_FAILED,
                        message=(
                            "tool completed but durable replay state could not be committed"
                        ),
                        retryable=False,
                    )
                    return self._to_call_result(uncertain)
            else:
                self._abandon_replay(replay_decision)
        return call_result

    @staticmethod
    def _timeout_seconds(definition: ToolDefinition, deadline_epoch_ms: int | None) -> float:
        timeout_ms = float(definition.default_timeout_ms)
        if deadline_epoch_ms is not None:
            remaining_ms = deadline_epoch_ms - time.time() * 1000.0
            timeout_ms = min(timeout_ms, remaining_ms)
        return max(0.0, timeout_ms / 1000.0)

    @staticmethod
    def _is_replayable_terminal(response: ToolResponseEnvelope[Any]) -> bool:
        if response.status in {
            ToolExecutionStatus.SUCCESS,
            ToolExecutionStatus.PARTIAL,
        }:
            return True
        return (
            response.status is ToolExecutionStatus.FAILED
            and response.error is not None
            and response.error.retryable is False
        )

    @staticmethod
    def _replay_identity(
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
    ) -> tuple[str, str]:
        context = arguments.get("context")
        if not isinstance(context, Mapping):
            raise ToolReplayLedgerError("tool replay context is missing")
        required = {
            "case_id": context.get("case_id"),
            "caller": context.get("caller"),
            "idempotency_key": context.get("idempotency_key"),
        }
        if any(
            not isinstance(value, str) or not value.strip()
            for value in required.values()
        ):
            raise ToolReplayLedgerError("tool replay identity is incomplete")
        scope_payload = {
            "tool_name": definition.name,
            "tool_version": definition.version,
            **required,
        }
        semantic_arguments = deepcopy(dict(arguments))
        semantic_context = dict(context)
        for volatile in (
            "request_id",
            "trace_id",
            "requested_at",
            "deadline_epoch_ms",
        ):
            semantic_context.pop(volatile, None)
        semantic_arguments["context"] = semantic_context
        try:
            scope_json = json.dumps(
                scope_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            request_json = json.dumps(
                {
                    "tool_name": definition.name,
                    "tool_version": definition.version,
                    "arguments": semantic_arguments,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ToolReplayLedgerError(
                "tool replay identity is not JSON-compatible"
            ) from exc
        return (
            sha256(scope_json.encode("utf-8")).hexdigest(),
            sha256(request_json.encode("utf-8")).hexdigest(),
        )

    def _replay_call_result(
        self,
        decision: ToolReplayDecision,
        *,
        definition: ToolDefinition,
        request_id: str,
        trace_id: str,
    ) -> McpCallResult:
        if decision.response is None:
            raise ToolReplayLedgerError("replay decision has no stored response")
        structured = deepcopy(dict(decision.response))
        envelope_fields = {
            "request_id",
            "trace_id",
            "tool_name",
            "tool_version",
            "status",
            "result",
            "artifacts",
            "metrics",
            "warnings",
            "error",
            "started_at",
            "finished_at",
        }
        if set(structured) != envelope_fields:
            raise ToolReplayLedgerError("stored replay response has invalid fields")
        if (
            structured.get("tool_name") != definition.name
            or structured.get("tool_version") != definition.version
        ):
            raise ToolReplayLedgerError("stored replay response identity is invalid")
        if structured.get("status") not in {"SUCCESS", "PARTIAL", "FAILED"}:
            raise ToolReplayLedgerError("stored replay response is not reusable")
        if structured.get("status") == "FAILED":
            error = structured.get("error")
            if not isinstance(error, Mapping) or error.get("retryable") is not False:
                raise ToolReplayLedgerError(
                    "stored retryable failure cannot be replayed"
                )
        structured["request_id"] = request_id
        structured["trace_id"] = trace_id
        return self._call_result_from_structured(structured, replay=True)

    def _abandon_replay(self, decision: ToolReplayDecision | None) -> None:
        if decision is None or self._replay_ledger is None:
            return
        try:
            self._replay_ledger.abandon(decision)
        except ToolReplayLedgerError:
            # A failed execution is already being returned.  An expired
            # read-only claim may be reclaimed; a write claim fails closed to
            # UNCERTAIN before another handler can run.
            pass

    def _mark_replay_uncertain(self, decision: ToolReplayDecision | None) -> None:
        if decision is None or self._replay_ledger is None:
            return
        try:
            self._replay_ledger.mark_uncertain(decision)
        except ToolReplayLedgerError:
            # A later write-like begin converts an expired PENDING record to
            # UNCERTAIN before any handler is allowed to execute again.
            pass

    def _failure_from_arguments(
        self,
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
        code: ErrorCode,
        message: str,
        *,
        field_path: str | None = None,
    ) -> ToolResponseEnvelope[Any]:
        context = arguments.get("context")
        context_mapping = context if isinstance(context, Mapping) else {}
        fingerprint = sha256(
            json.dumps(arguments, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        request_id = context_mapping.get("request_id")
        trace_id = context_mapping.get("trace_id")
        if not isinstance(request_id, str) or not request_id.strip():
            request_id = f"mcp-invalid-{fingerprint}"
        if not isinstance(trace_id, str) or not trace_id.strip():
            trace_id = f"mcp-invalid-trace-{fingerprint}"
        return self._failure(
            definition,
            request_id=request_id,
            trace_id=trace_id,
            code=code,
            message=message,
            retryable=False,
            field_path=field_path,
        )

    @staticmethod
    def _failure(
        definition: ToolDefinition,
        *,
        request_id: str,
        trace_id: str,
        code: ErrorCode,
        message: str,
        retryable: bool,
        field_path: str | None = None,
    ) -> ToolResponseEnvelope[Any]:
        now = _utc_now()
        return ToolResponseEnvelope(
            request_id=request_id,
            trace_id=trace_id,
            tool_name=definition.name,
            tool_version=definition.version,
            status=ToolExecutionStatus.FAILED,
            result=None,
            artifacts=(),
            metrics=(),
            warnings=(),
            error=ErrorDetail(
                code=code,
                message=message,
                retryable=retryable,
                field_path=field_path,
            ),
            started_at=now,
            finished_at=now,
        )

    @classmethod
    def _to_call_result(cls, response: ToolResponseEnvelope[Any]) -> McpCallResult:
        structured = to_mcp_safe_primitive(response)
        return cls._call_result_from_structured(structured, replay=False)

    @staticmethod
    def _call_result_from_structured(
        structured: Mapping[str, Any],
        *,
        replay: bool,
    ) -> McpCallResult:
        payload = deepcopy(dict(structured))
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        meta = {
            "com.turninqaq/requestId": payload["request_id"],
            "com.turninqaq/traceId": payload["trace_id"],
            "com.turninqaq/toolVersion": payload["tool_version"],
        }
        if replay:
            meta["com.turninqaq/idempotentReplay"] = True
        return McpCallResult(
            content=({"type": "text", "text": encoded},),
            structured_content=payload,
            is_error=payload["status"] == ToolExecutionStatus.FAILED.value,
            meta=meta,
        )
