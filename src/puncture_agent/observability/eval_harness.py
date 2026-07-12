"""End-to-end agent evaluation harness with explicit pass/fail diagnostics.

Preserves the original EvalCase / metric semantics while extending diagnostics,
tool-parameter predicates, recovery checks, RAG current-version/ACL signals,
and baseline-vs-candidate regression reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence

from puncture_agent.agent.runtime import GraphRuntime
from puncture_agent.agent.state import AgentState
from puncture_agent.observability.metrics import (
    METRICS_SCHEMA_VERSION,
    acl_violation_rate,
    active_version_hit_rate,
    percentile,
    tool_parameter_validity_rate,
    tool_selection_precision,
    tool_selection_recall,
)

EVAL_REPORT_SCHEMA_VERSION = "eval-report-v1"
REGRESSION_REPORT_SCHEMA_VERSION = "eval-regression-v1"


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    expected_task_type: str
    expected_terminal_status: str
    required_nodes: tuple[str, ...] = ()
    forbidden_nodes: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    minimum_citations: int = 0
    agent_case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    planning_constraints: dict[str, Any] = field(default_factory=dict)

    def build_state(self) -> AgentState:
        return AgentState(
            user_query=self.query,
            case_id=self.agent_case_id,
            metadata=dict(self.metadata),
            planning_constraints=dict(self.planning_constraints),
        )


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    passed: bool
    routing_correct: bool
    terminal_status_correct: bool
    required_node_coverage: float
    forbidden_node_violation_count: int
    tool_recall: float
    citation_requirement_met: bool
    schema_valid: bool
    failures: tuple[str, ...]
    observed: dict[str, Any]
    # Extended diagnostics (defaults preserve older callers / serialization).
    tool_precision: float = 1.0
    tool_parameter_validity: float = 1.0
    forbidden_tool_violation_count: int = 0
    recovery_correct: bool | None = None
    current_version_hit: bool | None = None
    acl_violation_count: int = 0
    trace_id: str | None = None
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalReport:
    case_count: int
    passed_case_count: int
    metrics: dict[str, float]
    cases: tuple[EvalCaseResult, ...]
    schema_version: str = EVAL_REPORT_SCHEMA_VERSION
    metrics_schema_version: str = METRICS_SCHEMA_VERSION
    dataset_version: str | None = None
    release_blocked: bool = False
    release_block_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metrics_schema_version": self.metrics_schema_version,
            "dataset_version": self.dataset_version,
            "case_count": self.case_count,
            "passed_case_count": self.passed_case_count,
            "metrics": dict(self.metrics),
            "release_blocked": self.release_blocked,
            "release_block_reasons": list(self.release_block_reasons),
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


@dataclass(frozen=True)
class RegressionCaseDelta:
    case_id: str
    classification: str  # improved | regressed | unchanged | new | removed
    baseline_passed: bool | None
    candidate_passed: bool | None
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionReport:
    schema_version: str
    baseline_dataset_version: str | None
    candidate_dataset_version: str | None
    improved_case_ids: tuple[str, ...]
    regressed_case_ids: tuple[str, ...]
    unchanged_case_ids: tuple[str, ...]
    new_case_ids: tuple[str, ...]
    removed_case_ids: tuple[str, ...]
    metric_deltas: dict[str, float]
    safety_regression: bool
    release_blocked: bool
    release_block_reasons: tuple[str, ...]
    cases: tuple[RegressionCaseDelta, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline_dataset_version": self.baseline_dataset_version,
            "candidate_dataset_version": self.candidate_dataset_version,
            "improved_case_ids": list(self.improved_case_ids),
            "regressed_case_ids": list(self.regressed_case_ids),
            "unchanged_case_ids": list(self.unchanged_case_ids),
            "new_case_ids": list(self.new_case_ids),
            "removed_case_ids": list(self.removed_case_ids),
            "metric_deltas": dict(self.metric_deltas),
            "safety_regression": self.safety_regression,
            "release_blocked": self.release_blocked,
            "release_block_reasons": list(self.release_block_reasons),
            "cases": [item.to_dict() for item in self.cases],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)


def _contains_node(visited_nodes: list[str], expected: str) -> bool:
    """Allow task cases to name a fully-qualified node or its terminal segment."""

    return expected in visited_nodes or any(
        visited == expected or visited.endswith(f".{expected}") for visited in visited_nodes
    )


def _validate_report_schema(report: dict[str, Any]) -> bool:
    required = {"report_version", "status", "case_id"}
    return isinstance(report, dict) and required.issubset(report)


def _document_ids_from_state(state: AgentState) -> list[str]:
    ids: list[str] = []
    for document in state.retrieved_documents:
        if not isinstance(document, Mapping):
            continue
        for key in ("document_id", "doc_id", "id"):
            value = document.get(key)
            if isinstance(value, str) and value:
                ids.append(value)
                break
    return ids


def _chunk_ids_from_state(state: AgentState) -> list[str]:
    ids: list[str] = []
    for document in state.retrieved_documents:
        if not isinstance(document, Mapping):
            continue
        for key in ("chunk_id", "id"):
            value = document.get(key)
            if isinstance(value, str) and value and key == "chunk_id":
                ids.append(value)
                break
    return ids


def evaluate_case(case: EvalCase, state: AgentState) -> EvalCaseResult:
    failures: list[str] = []
    diagnostics: list[str] = []
    routing_correct = state.task_type == case.expected_task_type
    if not routing_correct:
        failures.append(
            f"task_type expected {case.expected_task_type}, observed {state.task_type}"
        )

    terminal_status_correct = state.status == case.expected_terminal_status
    if not terminal_status_correct:
        failures.append(
            "terminal status expected "
            f"{case.expected_terminal_status}, observed {state.status}"
        )

    required_hits = sum(
        _contains_node(state.visited_nodes, node) for node in case.required_nodes
    )
    required_coverage = (
        required_hits / len(case.required_nodes) if case.required_nodes else 1.0
    )
    if required_coverage < 1.0:
        missing = [
            node
            for node in case.required_nodes
            if not _contains_node(state.visited_nodes, node)
        ]
        failures.append(f"required nodes not visited: {missing}")

    forbidden_hits = sum(
        _contains_node(state.visited_nodes, node) for node in case.forbidden_nodes
    )
    if forbidden_hits:
        failures.append("one or more forbidden nodes were visited")

    called_tools = {item.get("tool_name") for item in state.tool_calls}
    called_tools_clean = {name for name in called_tools if name}
    expected_tools = set(case.expected_tools)
    tool_recall = tool_selection_recall(called_tools_clean, expected_tools)
    tool_precision = tool_selection_precision(called_tools_clean, expected_tools)
    if tool_recall < 1.0:
        failures.append(f"expected tools not called: {sorted(expected_tools - called_tools_clean)}")

    forbidden_tools = {
        str(item)
        for item in (case.metadata.get("forbidden_tools") or [])
        if item
    }
    forbidden_tool_hits = sorted(called_tools_clean.intersection(forbidden_tools))
    if forbidden_tool_hits:
        failures.append(f"forbidden tools were called: {forbidden_tool_hits}")

    predicates = list(case.metadata.get("expected_tool_argument_predicates") or [])
    tool_param_validity = tool_parameter_validity_rate(state.tool_calls, predicates)
    if predicates and tool_param_validity < 1.0:
        failures.append("one or more tool argument predicates failed")
        diagnostics.append(
            f"tool parameter validity={tool_param_validity:.3f} predicates={len(predicates)}"
        )

    citation_ok = len(state.citations) >= case.minimum_citations
    if not citation_ok:
        failures.append(
            f"citations expected >= {case.minimum_citations}, observed {len(state.citations)}"
        )

    schema_valid = _validate_report_schema(state.final_report)
    if not schema_valid:
        failures.append("final_report does not satisfy the minimum schema")

    expected_error = case.metadata.get("expected_error_code")
    if expected_error:
        observed_codes = {
            str(item.get("code"))
            for item in state.errors
            if isinstance(item, Mapping) and item.get("code")
        }
        # Also inspect tool call errors when present.
        for call in state.tool_calls:
            error = call.get("error") if isinstance(call, Mapping) else None
            if isinstance(error, Mapping) and error.get("code"):
                observed_codes.add(str(error["code"]))
        if str(expected_error) not in observed_codes:
            failures.append(
                f"expected error code {expected_error!r} not observed; got {sorted(observed_codes)}"
            )

    recovery_correct: bool | None = None
    if case.metadata.get("is_recovery_case"):
        min_retry = case.metadata.get("expected_retry_count_min")
        max_retry = case.metadata.get("expected_retry_count_max")
        recovery_correct = terminal_status_correct
        if min_retry is not None and state.retry_count < int(min_retry):
            recovery_correct = False
            failures.append(
                f"retry_count expected >= {min_retry}, observed {state.retry_count}"
            )
        if max_retry is not None and state.retry_count > int(max_retry):
            recovery_correct = False
            failures.append(
                f"retry_count expected <= {max_retry}, observed {state.retry_count}"
            )
        if recovery_correct:
            diagnostics.append(
                f"recovery case passed with retry_count={state.retry_count}"
            )

    expected_docs = [
        str(item)
        for item in (case.metadata.get("expected_relevant_document_ids") or [])
        if item
    ]
    retrieved_docs = _document_ids_from_state(state)
    current_version_hit: bool | None = None
    expected_version = case.metadata.get("expected_document_version")
    if expected_version:
        versions = []
        for document in state.retrieved_documents:
            if isinstance(document, Mapping):
                version = document.get("version") or document.get("document_version")
                if version is not None:
                    versions.append(str(version))
        current_version_hit = str(expected_version) in versions or any(
            str(expected_version) == str(doc_id) for doc_id in retrieved_docs
        )
        # Prefer explicit metadata set by RAG adapters when present.
        if "current_version_hit" in state.metadata:
            current_version_hit = bool(state.metadata.get("current_version_hit"))
        if not current_version_hit:
            failures.append(
                f"required document version {expected_version!r} was not retrieved"
            )

    acl_violation_count = int(state.metadata.get("acl_violation_count") or 0)
    if acl_violation_count < 0:
        acl_violation_count = 0
    if acl_violation_count > 0:
        failures.append(f"ACL violations observed: {acl_violation_count}")

    if expected_docs and retrieved_docs:
        missing_docs = sorted(set(expected_docs) - set(retrieved_docs))
        if missing_docs:
            diagnostics.append(f"relevant documents not retrieved: {missing_docs}")

    trace_id = state.metadata.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id.strip():
        trace_id = None

    if failures:
        diagnostics.extend(failures)

    return EvalCaseResult(
        case_id=case.case_id,
        passed=not failures,
        routing_correct=routing_correct,
        terminal_status_correct=terminal_status_correct,
        required_node_coverage=required_coverage,
        forbidden_node_violation_count=forbidden_hits,
        tool_recall=tool_recall,
        citation_requirement_met=citation_ok,
        schema_valid=schema_valid,
        failures=tuple(failures),
        observed={
            "task_type": state.task_type,
            "status": state.status,
            "verification_status": state.verification_status,
            "visited_nodes": list(state.visited_nodes),
            "called_tools": sorted(name for name in called_tools_clean),
            "citation_count": len(state.citations),
            "retry_count": state.retry_count,
            "retrieved_document_ids": retrieved_docs,
            "retrieved_chunk_ids": _chunk_ids_from_state(state),
            "error_codes": sorted(
                {
                    str(item.get("code"))
                    for item in state.errors
                    if isinstance(item, Mapping) and item.get("code")
                }
            ),
            "trace_id": trace_id,
        },
        tool_precision=tool_precision,
        tool_parameter_validity=tool_param_validity,
        forbidden_tool_violation_count=len(forbidden_tool_hits),
        recovery_correct=recovery_correct,
        current_version_hit=current_version_hit,
        acl_violation_count=acl_violation_count,
        trace_id=trace_id,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


def _aggregate_metrics(results: Sequence[EvalCaseResult]) -> dict[str, float]:
    latencies = [
        float(item.observed["latency_ms"])
        for item in results
        if isinstance(item.observed.get("latency_ms"), (int, float))
    ]
    step_counts = [
        float(len(item.observed.get("visited_nodes") or [])) for item in results
    ]
    version_cases = [
        {
            "current_version_hit": item.current_version_hit,
            "required_version": "set" if item.current_version_hit is not None else None,
        }
        for item in results
        if item.current_version_hit is not None
    ]
    recovery_cases = [
        {
            "is_recovery_case": True,
            "recovery_correct": bool(item.recovery_correct),
        }
        for item in results
        if item.recovery_correct is not None
    ]
    total_chunks = sum(len(item.observed.get("retrieved_chunk_ids") or []) for item in results)
    total_acl = sum(item.acl_violation_count for item in results)

    metrics: dict[str, float] = {
        "task_success_rate": fmean(float(item.passed) for item in results),
        "routing_accuracy": fmean(float(item.routing_correct) for item in results),
        "terminal_status_accuracy": fmean(
            float(item.terminal_status_correct) for item in results
        ),
        "required_node_coverage": fmean(
            item.required_node_coverage for item in results
        ),
        "forbidden_node_violation_rate": fmean(
            float(item.forbidden_node_violation_count > 0) for item in results
        ),
        "forbidden_tool_violation_rate": fmean(
            float(item.forbidden_tool_violation_count > 0) for item in results
        ),
        "tool_recall": fmean(item.tool_recall for item in results),
        "tool_precision": fmean(item.tool_precision for item in results),
        "tool_parameter_validity_rate": fmean(
            item.tool_parameter_validity for item in results
        ),
        "citation_pass_rate": fmean(
            float(item.citation_requirement_met) for item in results
        ),
        "report_schema_validity_rate": fmean(
            float(item.schema_valid) for item in results
        ),
        "active_version_hit_rate": active_version_hit_rate(version_cases),
        "acl_violation_rate": acl_violation_rate(total_acl, max(total_chunks, total_acl)),
        "retry_recovery_rate": (
            fmean(float(item["recovery_correct"]) for item in recovery_cases)
            if recovery_cases
            else 1.0
        ),
        "mean_steps": fmean(step_counts) if step_counts else 0.0,
        "p95_steps": percentile(step_counts, 95) if step_counts else 0.0,
    }
    if latencies:
        metrics["mean_latency_ms"] = fmean(latencies)
        metrics["p95_latency_ms"] = percentile(latencies, 95)
    return metrics


def evaluate_release_gates(report: EvalReport) -> tuple[bool, tuple[str, ...]]:
    """Return whether release must be blocked and the reasons why."""

    reasons: list[str] = []
    metrics = report.metrics
    if metrics.get("forbidden_node_violation_rate", 0.0) > 0.0:
        reasons.append("forbidden_node_violation_rate > 0")
    if metrics.get("forbidden_tool_violation_rate", 0.0) > 0.0:
        reasons.append("forbidden_tool_violation_rate > 0")
    if metrics.get("acl_violation_rate", 0.0) > 0.0:
        reasons.append("acl_violation_rate > 0")
    if metrics.get("report_schema_validity_rate", 1.0) < 1.0:
        reasons.append("report_schema_validity_rate < 1")
    # Failed cases without diagnostics or trace linkage block release.
    for case in report.cases:
        if case.passed:
            continue
        if not case.failures and not case.diagnostics:
            reasons.append(f"case {case.case_id} failed without diagnostics")
        if not case.trace_id and not case.observed.get("trace_id"):
            # Trace is required for failed cases when instrumentation is present;
            # mock harness runs may omit it. Only block when explicitly expected.
            if case.observed.get("require_trace_on_failure"):
                reasons.append(f"case {case.case_id} failed without trace_id")
    return (bool(reasons), tuple(dict.fromkeys(reasons)))


class AgentEvalHarness:
    """Run isolated evaluation cases against a fresh runtime per case."""

    def __init__(self, runtime_factory: Callable[[], GraphRuntime]) -> None:
        self.runtime_factory = runtime_factory

    def run(
        self,
        cases: Iterable[EvalCase],
        *,
        dataset_version: str | None = None,
    ) -> EvalReport:
        case_list = list(cases)
        if not case_list:
            raise ValueError("At least one EvalCase is required")

        results: list[EvalCaseResult] = []
        for case in case_list:
            runtime = self.runtime_factory()
            state = runtime.run(case.build_state())
            results.append(evaluate_case(case, state))

        report = EvalReport(
            case_count=len(results),
            passed_case_count=sum(item.passed for item in results),
            metrics=_aggregate_metrics(results),
            cases=tuple(results),
            dataset_version=dataset_version,
        )
        blocked, reasons = evaluate_release_gates(report)
        if blocked or reasons:
            report = EvalReport(
                case_count=report.case_count,
                passed_case_count=report.passed_case_count,
                metrics=report.metrics,
                cases=report.cases,
                schema_version=report.schema_version,
                metrics_schema_version=report.metrics_schema_version,
                dataset_version=report.dataset_version,
                release_blocked=blocked,
                release_block_reasons=reasons,
            )
        return report


def compare_reports(
    baseline: EvalReport,
    candidate: EvalReport,
) -> RegressionReport:
    """Classify per-case improvements/regressions and metric deltas."""

    baseline_by_id = {item.case_id: item for item in baseline.cases}
    candidate_by_id = {item.case_id: item for item in candidate.cases}
    all_ids = sorted(set(baseline_by_id) | set(candidate_by_id))

    deltas: list[RegressionCaseDelta] = []
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    new_ids: list[str] = []
    removed: list[str] = []

    for case_id in all_ids:
        base = baseline_by_id.get(case_id)
        cand = candidate_by_id.get(case_id)
        if base is None and cand is not None:
            classification = "new"
            new_ids.append(case_id)
            reasons = ("case present only in candidate",)
        elif base is not None and cand is None:
            classification = "removed"
            removed.append(case_id)
            reasons = ("case present only in baseline",)
        else:
            assert base is not None and cand is not None
            if base.passed and not cand.passed:
                classification = "regressed"
                regressed.append(case_id)
                reasons = cand.failures or ("candidate failed previously passing case",)
            elif not base.passed and cand.passed:
                classification = "improved"
                improved.append(case_id)
                reasons = ("candidate now passes previously failing case",)
            else:
                classification = "unchanged"
                unchanged.append(case_id)
                reasons = ()
        deltas.append(
            RegressionCaseDelta(
                case_id=case_id,
                classification=classification,
                baseline_passed=None if base is None else base.passed,
                candidate_passed=None if cand is None else cand.passed,
                reasons=tuple(reasons),
            )
        )

    metric_keys = sorted(set(baseline.metrics) | set(candidate.metrics))
    metric_deltas = {
        key: float(candidate.metrics.get(key, 0.0)) - float(baseline.metrics.get(key, 0.0))
        for key in metric_keys
    }

    block_reasons: list[str] = []
    safety_keys = (
        "forbidden_node_violation_rate",
        "forbidden_tool_violation_rate",
        "acl_violation_rate",
    )
    safety_regression = False
    for key in safety_keys:
        if metric_deltas.get(key, 0.0) > 0.0:
            safety_regression = True
            block_reasons.append(f"safety metric worsened: {key}")
    if regressed:
        # Safety/security related regressions always block; general quality
        # regressions are listed and also block by default for contract suites.
        block_reasons.append(f"regressed cases: {regressed}")
    if candidate.release_blocked:
        block_reasons.extend(candidate.release_block_reasons)
        safety_regression = True

    return RegressionReport(
        schema_version=REGRESSION_REPORT_SCHEMA_VERSION,
        baseline_dataset_version=baseline.dataset_version,
        candidate_dataset_version=candidate.dataset_version,
        improved_case_ids=tuple(improved),
        regressed_case_ids=tuple(regressed),
        unchanged_case_ids=tuple(unchanged),
        new_case_ids=tuple(new_ids),
        removed_case_ids=tuple(removed),
        metric_deltas=metric_deltas,
        safety_regression=safety_regression,
        release_blocked=bool(block_reasons),
        release_block_reasons=tuple(dict.fromkeys(block_reasons)),
        cases=tuple(deltas),
    )


def load_report(path: str) -> EvalReport:
    """Load a previously serialized aggregate report."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("report root must be an object")
    cases = tuple(
        EvalCaseResult(
            case_id=str(item["case_id"]),
            passed=bool(item["passed"]),
            routing_correct=bool(item.get("routing_correct", False)),
            terminal_status_correct=bool(item.get("terminal_status_correct", False)),
            required_node_coverage=float(item.get("required_node_coverage", 0.0)),
            forbidden_node_violation_count=int(
                item.get("forbidden_node_violation_count", 0)
            ),
            tool_recall=float(item.get("tool_recall", 0.0)),
            citation_requirement_met=bool(item.get("citation_requirement_met", False)),
            schema_valid=bool(item.get("schema_valid", False)),
            failures=tuple(item.get("failures") or ()),
            observed=dict(item.get("observed") or {}),
            tool_precision=float(item.get("tool_precision", 1.0)),
            tool_parameter_validity=float(item.get("tool_parameter_validity", 1.0)),
            forbidden_tool_violation_count=int(
                item.get("forbidden_tool_violation_count", 0)
            ),
            recovery_correct=item.get("recovery_correct"),
            current_version_hit=item.get("current_version_hit"),
            acl_violation_count=int(item.get("acl_violation_count", 0)),
            trace_id=item.get("trace_id"),
            diagnostics=tuple(item.get("diagnostics") or ()),
        )
        for item in payload.get("cases") or []
    )
    return EvalReport(
        case_count=int(payload.get("case_count", len(cases))),
        passed_case_count=int(
            payload.get("passed_case_count", sum(item.passed for item in cases))
        ),
        metrics={str(k): float(v) for k, v in dict(payload.get("metrics") or {}).items()},
        cases=cases,
        schema_version=str(payload.get("schema_version") or EVAL_REPORT_SCHEMA_VERSION),
        metrics_schema_version=str(
            payload.get("metrics_schema_version") or METRICS_SCHEMA_VERSION
        ),
        dataset_version=payload.get("dataset_version"),
        release_blocked=bool(payload.get("release_blocked", False)),
        release_block_reasons=tuple(payload.get("release_block_reasons") or ()),
    )
