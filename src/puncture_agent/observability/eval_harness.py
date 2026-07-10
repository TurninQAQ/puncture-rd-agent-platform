"""End-to-end agent evaluation harness with explicit pass/fail diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any, Callable, Iterable

from puncture_agent.agent.runtime import GraphRuntime
from puncture_agent.agent.state import AgentState


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalReport:
    case_count: int
    passed_case_count: int
    metrics: dict[str, float]
    cases: tuple[EvalCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "passed_case_count": self.passed_case_count,
            "metrics": dict(self.metrics),
            "cases": [case.to_dict() for case in self.cases],
        }


def _contains_node(visited_nodes: list[str], expected: str) -> bool:
    """Allow task cases to name a fully-qualified node or its terminal segment."""

    return expected in visited_nodes or any(
        visited == expected or visited.endswith(f".{expected}") for visited in visited_nodes
    )


def _validate_report_schema(report: dict[str, Any]) -> bool:
    required = {"report_version", "status", "case_id"}
    return isinstance(report, dict) and required.issubset(report)


def evaluate_case(case: EvalCase, state: AgentState) -> EvalCaseResult:
    failures: list[str] = []
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
    expected_tools = set(case.expected_tools)
    tool_recall = (
        len(called_tools.intersection(expected_tools)) / len(expected_tools)
        if expected_tools
        else 1.0
    )
    if tool_recall < 1.0:
        failures.append(f"expected tools not called: {sorted(expected_tools - called_tools)}")

    citation_ok = len(state.citations) >= case.minimum_citations
    if not citation_ok:
        failures.append(
            f"citations expected >= {case.minimum_citations}, observed {len(state.citations)}"
        )

    schema_valid = _validate_report_schema(state.final_report)
    if not schema_valid:
        failures.append("final_report does not satisfy the minimum schema")

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
            "called_tools": sorted(name for name in called_tools if name),
            "citation_count": len(state.citations),
            "retry_count": state.retry_count,
        },
    )


class AgentEvalHarness:
    """Run isolated evaluation cases against a fresh runtime per case."""

    def __init__(self, runtime_factory: Callable[[], GraphRuntime]) -> None:
        self.runtime_factory = runtime_factory

    def run(self, cases: Iterable[EvalCase]) -> EvalReport:
        case_list = list(cases)
        if not case_list:
            raise ValueError("At least one EvalCase is required")

        results: list[EvalCaseResult] = []
        for case in case_list:
            runtime = self.runtime_factory()
            state = runtime.run(case.build_state())
            results.append(evaluate_case(case, state))

        metrics = {
            "task_success_rate": fmean(float(item.passed) for item in results),
            "routing_accuracy": fmean(
                float(item.routing_correct) for item in results
            ),
            "terminal_status_accuracy": fmean(
                float(item.terminal_status_correct) for item in results
            ),
            "required_node_coverage": fmean(
                item.required_node_coverage for item in results
            ),
            "forbidden_node_violation_rate": fmean(
                float(item.forbidden_node_violation_count > 0) for item in results
            ),
            "tool_recall": fmean(item.tool_recall for item in results),
            "citation_pass_rate": fmean(
                float(item.citation_requirement_met) for item in results
            ),
            "report_schema_validity_rate": fmean(
                float(item.schema_valid) for item in results
            ),
        }
        return EvalReport(
            case_count=len(results),
            passed_case_count=sum(item.passed for item in results),
            metrics=metrics,
            cases=tuple(results),
        )
