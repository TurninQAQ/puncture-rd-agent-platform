"""Production adapters for model-backed parsing and enterprise RAG nodes.

Algorithm and safety facts remain in the frozen tools and deterministic
verifier.  The model may classify an explicitly supplied request and the RAG
service may provide compact citations, but neither adapter is allowed to write
algorithm results or choose an unauthorized graph edge.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from puncture_agent.model_gateway.client import ModelGateway, ModelGatewayError
from puncture_agent.model_gateway.models import ChatMessage, ModelRequest
from puncture_agent.rag.client import RagService
from puncture_agent.rag.errors import RagServiceError
from puncture_agent.rag.models import RetrievalRequest

from .nodes import ToolExecutor, build_mock_handlers
from .runtime import NodeContext, NodeOutcome
from .state import AgentState, TaskType
from .tool_bridge import DEFAULT_TOOL_BRIDGE_POLICY, ToolBridgePolicy


_TASK_TOOL_PLANS = MappingProxyType(
    {
        TaskType.DATA_MODEL_VALIDATION: (
            "inspect_case_metadata",
            "validate_label_schema",
            "run_segmentation",
            "validate_segmentation_result",
            "extract_skin_surface",
        ),
        TaskType.PLANNING_SAFETY: (
            "generate_candidate_paths",
            "evaluate_path_safety",
            "evaluate_intraoperative_risk",
            "verify_skin_penetration",
        ),
    }
)

_TASK_RAG_MODULES = MappingProxyType(
    {
        TaskType.DATA_MODEL_VALIDATION: ("data_validation", "segmentation"),
        TaskType.PLANNING_SAFETY: ("path_planning", "safety_evaluation"),
    }
)

_RAG_INCOMPLETE_EVIDENCE = "RAG_INCOMPLETE_EVIDENCE"

_REQUEST_PLAN_SCHEMA: Mapping[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "task_type": {
                "type": "string",
                "enum": [
                    TaskType.UNKNOWN,
                    TaskType.DATA_MODEL_VALIDATION,
                    TaskType.PLANNING_SAFETY,
                ],
            },
            "case_id": {"type": "string"},
            "tool_plan": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 10,
            },
            "input_format": {
                "type": "string",
                "enum": ["AUTO", "MCS", "NIFTI"],
            },
            "run_segmentation": {"type": "boolean"},
            "extract_skin_surface": {"type": "boolean"},
        },
        "required": [
            "task_type",
            "case_id",
            "tool_plan",
            "input_format",
            "run_segmentation",
            "extract_skin_surface",
        ],
}


class ProductionNodeConfigurationError(ValueError):
    """Raised when production dependencies or policies are incomplete."""


class StructuredRequestPlanner(Protocol):
    def plan(self, state: AgentState) -> Mapping[str, Any] | None: ...


class KnowledgeRetriever(Protocol):
    def retrieve(self, state: AgentState) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...


_TEST_ONLY_METADATA_KEYS = frozenset(
    {
        "fail_tool_once",
        "fail_tool_always",
        "fail_tool_non_retryable",
        "failed_tools_once",
        "force_case_missing",
        "force_geometry_mismatch",
        "force_label_schema_error",
        "force_empty_segmentation",
        "force_no_feasible_path",
        "missing_required_artifacts",
        "model_gateway_metadata",
        "use_mock_artifacts",
    }
)


def _stable_id(prefix: str, *parts: Any) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode(
        "utf-8"
    )
    return f"{prefix}-{sha256(encoded).hexdigest()[:24]}"


def _case_ids_in_query(query: str) -> set[str]:
    return {
        match.group(0).replace("_", "-").replace(" ", "-").lower()
        for match in re.finditer(
            r"(?<![A-Za-z0-9])case[-_\s]?\d+(?![A-Za-z0-9])",
            query,
            flags=re.IGNORECASE,
        )
    }


class GatewayRequestPlanner:
    """Use the model gateway for bounded structured request classification."""

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        max_attempts: int = 2,
        request_metadata_provider: Callable[[AgentState], Mapping[str, Any]] | None = None,
    ) -> None:
        if not callable(getattr(gateway, "generate", None)):
            raise ProductionNodeConfigurationError("model gateway must expose generate()")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("max_attempts must be an integer")
        if not 1 <= max_attempts <= 2:
            raise ValueError("max_attempts must be 1 or 2")
        self.gateway = gateway
        self.max_attempts = max_attempts
        self.request_metadata_provider = request_metadata_provider

    def plan(self, state: AgentState) -> Mapping[str, Any] | None:
        messages = (
            ChatMessage(
                "system",
                "Classify the request into the fixed workflow schema. "
                "Only copy a case ID that is explicitly present in the user text. "
                "The tool plan is advisory and cannot authorize tools.",
            ),
            ChatMessage("user", state.user_query),
        )
        last_failure = "MODEL_STRUCTURED_OUTPUT_INVALID"
        for attempt in range(self.max_attempts):
            request_metadata: Mapping[str, Any] = {}
            if self.request_metadata_provider is not None:
                request_metadata = self.request_metadata_provider(state)
                if not isinstance(request_metadata, Mapping):
                    raise ProductionNodeConfigurationError(
                        "model request metadata provider must return an object"
                    )
            request = ModelRequest(
                request_id=_stable_id(
                    "agent-plan",
                    state.session_id,
                    state.user_query,
                    attempt,
                ),
                messages=messages
                + (
                    (
                        ChatMessage(
                            "user",
                            "The previous response was invalid. Return exactly the required JSON object.",
                        ),
                    )
                    if attempt
                    else ()
                ),
                response_schema=deepcopy(_REQUEST_PLAN_SCHEMA),
                temperature=0.0,
                max_tokens=384,
                metadata=dict(request_metadata),
            )
            try:
                response = self.gateway.generate(request)
                candidate = response.structured_output
                return self._validate_plan(candidate, state)
            except ModelGatewayError as exc:
                last_failure = exc.code
                if not exc.retryable or attempt + 1 >= self.max_attempts:
                    break
            except (TypeError, ValueError, KeyError):
                last_failure = "MODEL_STRUCTURED_OUTPUT_INVALID"
                if attempt + 1 >= self.max_attempts:
                    break

        state.metadata["model_parse_failed"] = True
        state.add_error(
            last_failure,
            "request classification did not produce a valid bounded structured result",
            retryable=False,
            details={"attempts": self.max_attempts},
        )
        return None

    @staticmethod
    def _validate_plan(candidate: Any, state: AgentState) -> Mapping[str, Any]:
        if not isinstance(candidate, Mapping):
            raise ValueError("structured_output must be an object")
        required = set(_REQUEST_PLAN_SCHEMA["required"])
        if set(candidate) != required:
            raise ValueError("structured_output fields do not match the request-plan schema")

        task_type = candidate["task_type"]
        if task_type not in {
            TaskType.UNKNOWN,
            TaskType.DATA_MODEL_VALIDATION,
            TaskType.PLANNING_SAFETY,
        }:
            raise ValueError("unsupported task_type")
        case_id = candidate["case_id"]
        if not isinstance(case_id, str):
            raise ValueError("case_id must be a string")
        normalized_case = case_id.strip().replace("_", "-").replace(" ", "-") or None
        if normalized_case:
            explicit_ids = _case_ids_in_query(state.user_query)
            if normalized_case.lower() not in explicit_ids:
                raise ValueError("model returned a case ID not present in the request")
            if state.case_id and state.case_id.lower() != normalized_case.lower():
                raise ValueError("model case ID conflicts with API case context")

        proposed_tools = candidate["tool_plan"]
        if (
            isinstance(proposed_tools, (str, bytes))
            or not isinstance(proposed_tools, list)
            or any(not isinstance(name, str) for name in proposed_tools)
        ):
            raise ValueError("tool_plan must be an array of strings")
        if len(proposed_tools) > 10 or len(proposed_tools) != len(set(proposed_tools)):
            raise ValueError("tool_plan must contain at most ten unique tools")
        allowed = set(_TASK_TOOL_PLANS.get(task_type, ()))
        if any(name not in allowed for name in proposed_tools):
            raise ValueError("model proposed a tool outside the fixed task catalog")

        input_format = candidate["input_format"]
        if input_format not in {"AUTO", "MCS", "NIFTI"}:
            raise ValueError("unsupported input_format")
        for flag in ("run_segmentation", "extract_skin_surface"):
            if not isinstance(candidate[flag], bool):
                raise ValueError(f"{flag} must be a boolean")
        return {
            "task_type": task_type,
            "case_id": normalized_case,
            "tool_plan": list(_TASK_TOOL_PLANS.get(task_type, ())),
            "model_proposed_tools": list(proposed_tools),
            "input_format": input_format,
            "run_segmentation": candidate["run_segmentation"],
            "extract_skin_surface": candidate["extract_skin_surface"],
        }


class RagKnowledgeRetriever:
    """Map the enterprise RAG contract to compact, model-safe graph state."""

    def __init__(
        self,
        service: RagService,
        *,
        access_scope_provider: Callable[[AgentState], tuple[str, ...]],
        top_k: int = 5,
    ) -> None:
        if not callable(getattr(service, "retrieve", None)):
            raise ProductionNodeConfigurationError("RAG service must expose retrieve()")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        self.service = service
        if not callable(access_scope_provider):
            raise ProductionNodeConfigurationError(
                "an authenticated RAG access-scope provider is required"
            )
        self.access_scope_provider = access_scope_provider
        self.top_k = top_k

    def retrieve(
        self,
        state: AgentState,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        modules = _TASK_RAG_MODULES.get(state.task_type, ())
        access_scopes = self.access_scope_provider(state)
        if isinstance(access_scopes, (str, bytes)) or not isinstance(
            access_scopes, tuple
        ):
            raise ValueError("access_scopes must be a sequence of scope strings")
        request = RetrievalRequest(
            request_id=_stable_id(
                "agent-rag",
                state.session_id,
                state.user_query,
                state.task_type,
            ),
            query=state.user_query,
            modules=tuple(modules),
            access_scopes=tuple(access_scopes),
            top_k=self.top_k,
            metadata_filters={"status": "active"},
        )
        response = self.service.retrieve(request)
        documents = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "module": chunk.module,
                "version": chunk.version,
                "section": chunk.section,
                "score": chunk.score,
                "rank": chunk.rank,
                "citation": chunk.citation,
            }
            for chunk in response.chunks
        ]
        citations = [
            {
                "document_id": chunk.document_id,
                "version": chunk.version,
                "section": chunk.section,
                "citation": chunk.citation,
            }
            for chunk in response.chunks
        ]
        return documents, citations


def _model_parse_node(
    state: AgentState,
    _: NodeContext,
    planner: StructuredRequestPlanner,
) -> NodeOutcome:
    plan = planner.plan(state)
    if plan is None:
        state.task_type = TaskType.UNKNOWN
        state.tool_plan = []
        return NodeOutcome(output={"status": "INVALID_MODEL_OUTPUT"})
    state.task_type = str(plan["task_type"])
    if not state.case_id and plan.get("case_id"):
        state.case_id = str(plan["case_id"])
    state.tool_plan = list(plan["tool_plan"])
    state.metadata["model_proposed_tools"] = list(plan["model_proposed_tools"])
    if plan["input_format"] != "AUTO":
        state.metadata["input_format"] = plan["input_format"]
    state.metadata["run_segmentation"] = plan["run_segmentation"]
    state.metadata["extract_skin_surface"] = plan["extract_skin_surface"]
    return NodeOutcome(
        output={
            "task_type": state.task_type,
            "case_id": state.case_id,
            "tool_plan": list(state.tool_plan),
        }
    )


def _rag_node(
    state: AgentState,
    _: NodeContext,
    retriever: KnowledgeRetriever,
) -> NodeOutcome:
    required_modules = tuple(_TASK_RAG_MODULES.get(state.task_type, ()))
    try:
        documents, citations = retriever.retrieve(state)
    except RagServiceError as exc:
        state.retrieved_documents = []
        state.citations = []
        state.metadata["rag_context_ready"] = False
        state.metadata["rag_required_modules"] = list(required_modules)
        state.metadata["rag_evidence_modules"] = []
        state.metadata["rag_missing_modules"] = list(required_modules)
        state.add_error(
            exc.code,
            "project-knowledge retrieval failed",
            retryable=exc.retryable,
        )
        return NodeOutcome(output={"document_count": 0, "error_code": exc.code})
    except Exception as exc:
        state.retrieved_documents = []
        state.citations = []
        state.metadata["rag_context_ready"] = False
        state.metadata["rag_required_modules"] = list(required_modules)
        state.metadata["rag_evidence_modules"] = []
        state.metadata["rag_missing_modules"] = list(required_modules)
        state.add_error(
            "RAG_PROTOCOL_ERROR",
            f"project-knowledge adapter rejected {type(exc).__name__}",
            retryable=False,
        )
        return NodeOutcome(output={"document_count": 0, "error_code": "RAG_PROTOCOL_ERROR"})
    state.retrieved_documents = documents
    state.citations = citations
    evidence_modules = sorted(
        {
            module
            for document in documents
            if isinstance(document, Mapping)
            and isinstance((module := document.get("module")), str)
            and module
        }
    )
    evidence_module_set = set(evidence_modules)
    missing_modules = [
        module for module in required_modules if module not in evidence_module_set
    ]
    state.metadata["rag_required_modules"] = list(required_modules)
    state.metadata["rag_evidence_modules"] = evidence_modules
    state.metadata["rag_missing_modules"] = missing_modules
    state.metadata["rag_context_ready"] = bool(documents) and not missing_modules
    if not documents:
        state.add_error(
            "RAG_NO_EVIDENCE",
            "no active project-knowledge evidence matched the request",
            retryable=False,
            details={"required_modules": list(required_modules)},
        )
        return NodeOutcome(
            output={"document_count": 0, "error_code": "RAG_NO_EVIDENCE"}
        )
    if missing_modules:
        state.add_error(
            _RAG_INCOMPLETE_EVIDENCE,
            "required project-knowledge module evidence is incomplete",
            retryable=False,
            details={
                "required_modules": list(required_modules),
                "missing_modules": list(missing_modules),
            },
        )
        return NodeOutcome(
            output={
                "document_count": len(documents),
                "error_code": _RAG_INCOMPLETE_EVIDENCE,
                "missing_modules": list(missing_modules),
            }
        )
    return NodeOutcome(
        output={
            "document_count": len(documents),
            "evidence_modules": evidence_modules,
        }
    )


def _resolve_production_case_context(
    state: AgentState,
    _: NodeContext,
    *,
    tool_policy: ToolBridgePolicy,
) -> NodeOutcome:
    """Validate supplied Artifact IDs without inventing development fixtures."""

    required_id_fields = {
        TaskType.DATA_MODEL_VALIDATION: ("ct", "raw_labels"),
        TaskType.PLANNING_SAFETY: ("ct", "skin", "skin_surface", "target"),
    }.get(state.task_type, ())
    missing = [
        name
        for name in required_id_fields
        if not isinstance(state.artifacts.get(name), str)
        or not str(state.artifacts.get(name)).strip()
    ]
    danger_masks = state.artifacts.get("danger_masks")
    if state.task_type == TaskType.PLANNING_SAFETY:
        if not isinstance(danger_masks, Mapping) or not danger_masks:
            missing.append("danger_masks")
        else:
            known_keys = {
                key for rule in tool_policy.danger_masks for key in rule.request_keys
            }
            for key, value in danger_masks.items():
                if (
                    not isinstance(key, str)
                    or key not in known_keys
                    or not isinstance(value, str)
                    or not value.strip()
                ):
                    missing.append(f"danger_masks.{key}")
            for rule in tool_policy.danger_masks:
                present = [key for key in rule.request_keys if key in danger_masks]
                if len(present) > 1:
                    missing.append(f"danger_masks.{rule.structure.lower()}.ambiguous")
                elif rule.required and not present:
                    missing.append(f"danger_masks.{rule.structure.lower()}")
    missing.extend(state.metadata.get("missing_required_artifacts", ()))
    if state.task_type in {
        TaskType.DATA_MODEL_VALIDATION,
        TaskType.PLANNING_SAFETY,
    } and state.metadata.get("rag_context_ready") is not True:
        missing.append("project_knowledge")
    state.metadata["missing_required_artifacts"] = sorted(set(missing))
    ready = bool(state.case_id) and not missing
    state.metadata["case_context_ready"] = ready
    return NodeOutcome(
        output={
            "case_context_ready": ready,
            "missing_artifacts": list(state.metadata["missing_required_artifacts"]),
        }
    )


def build_production_handlers(
    *,
    tool_executor: ToolExecutor,
    model_gateway: ModelGateway,
    rag_service: RagService,
    access_scope_provider: Callable[[AgentState], tuple[str, ...]],
    allow_test_controls: bool = False,
) -> Mapping[str, Any]:
    """Build the exact checked-in handler catalog with explicit dependencies."""

    if tool_executor is None or model_gateway is None or rag_service is None:
        raise ProductionNodeConfigurationError(
            "tool_executor, model_gateway, and rag_service must all be injected"
        )
    planner = GatewayRequestPlanner(
        model_gateway,
        request_metadata_provider=(
            (lambda state: dict(state.metadata.get("model_gateway_metadata", {})))
            if allow_test_controls
            else None
        ),
    )
    retriever = RagKnowledgeRetriever(
        rag_service,
        access_scope_provider=access_scope_provider,
    )
    handlers = build_mock_handlers(tool_executor)

    def parse_request(state: AgentState, context: NodeContext) -> NodeOutcome:
        forbidden = sorted(set(state.metadata).intersection(_TEST_ONLY_METADATA_KEYS))
        if forbidden and not allow_test_controls:
            state.task_type = TaskType.UNKNOWN
            state.tool_plan = []
            state.metadata["rejected_runtime_controls"] = forbidden
            state.add_error(
                "INVALID_ARGUMENT",
                "request metadata contains test-only runtime controls",
                retryable=False,
                details={"fields": forbidden},
            )
            return NodeOutcome(output={"status": "INVALID_RUNTIME_METADATA"})
        return _model_parse_node(state, context, planner)

    handlers["parse_request"] = parse_request
    handlers["retrieve_project_knowledge"] = lambda state, context: _rag_node(
        state, context, retriever
    )
    policy = getattr(tool_executor, "policy", DEFAULT_TOOL_BRIDGE_POLICY)
    if not isinstance(policy, ToolBridgePolicy):
        raise ProductionNodeConfigurationError(
            "tool executor policy must be a ToolBridgePolicy"
        )
    handlers["resolve_case_context"] = lambda state, context: (
        _resolve_production_case_context(
            state,
            context,
            tool_policy=policy,
        )
    )
    return MappingProxyType(handlers)


__all__ = [
    "GatewayRequestPlanner",
    "KnowledgeRetriever",
    "ProductionNodeConfigurationError",
    "RagKnowledgeRetriever",
    "StructuredRequestPlanner",
    "build_production_handlers",
]
