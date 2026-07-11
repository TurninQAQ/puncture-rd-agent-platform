from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent.nodes import DeterministicMockToolExecutor  # noqa: E402
from puncture_agent.agent.production_nodes import (  # noqa: E402
    GatewayRequestPlanner,
    RagKnowledgeRetriever,
    build_production_handlers,
)
from puncture_agent.agent.runtime import NodeContext  # noqa: E402
from puncture_agent.agent.state import AgentState, TaskType  # noqa: E402
from puncture_agent.model_gateway.mock_qwen import MockQwenGateway  # noqa: E402
from puncture_agent.rag.errors import RagServiceError  # noqa: E402
from puncture_agent.rag.models import (  # noqa: E402
    RetrievedChunk,
    RetrievalResponse,
)


def context(name: str) -> NodeContext:
    return NodeContext("puncture_agent_main", name, name, {})


def planning_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "task_type": TaskType.PLANNING_SAFETY,
        "case_id": "Case-701",
        "tool_plan": ["generate_candidate_paths"],
        "input_format": "AUTO",
        "run_segmentation": False,
        "extract_skin_surface": False,
    }
    payload.update(overrides)
    return payload


class FakeRagService:
    def __init__(self) -> None:
        self.requests = []

    def health(self):  # pragma: no cover - protocol completeness
        raise NotImplementedError

    def retrieve(self, request):
        self.requests.append(request)
        return RetrievalResponse(
            request_id=request.request_id,
            rewritten_query=request.query,
            chunks=(
                RetrievedChunk(
                    chunk_id="chunk-1",
                    document_id="planning-rule",
                    title="Planning rule",
                    module="planning",
                    version="3.0",
                    section="Needle constraints",
                    text="This text must not be copied into AgentState.",
                    score=0.97,
                    rank=1,
                    citation="planning-rule@3.0#Needle-constraints",
                ),
            ),
            retrieval_mode="hybrid",
            trace_id="rag-trace-1",
            latency_ms=1.0,
        )


class FailingRagService(FakeRagService):
    def retrieve(self, request):
        raise RagServiceError("RAG_TIMEOUT", "private backend detail", retryable=True)


class ModuleCoverageRagService(FakeRagService):
    def __init__(self, modules: tuple[str, ...]) -> None:
        super().__init__()
        self.modules = modules

    def retrieve(self, request):
        self.requests.append(request)
        chunks = tuple(
            RetrievedChunk(
                chunk_id=f"chunk-{index}",
                document_id=f"{module}-rule",
                title=f"{module} rule",
                module=module,
                version="3.0",
                section="Active rule",
                text="This text must not be copied into AgentState.",
                score=0.99 - index * 0.01,
                rank=index + 1,
                citation=f"{module}-rule@3.0#Active-rule",
            )
            for index, module in enumerate(self.modules)
        )
        return RetrievalResponse(
            request_id=request.request_id,
            rewritten_query=request.query,
            chunks=chunks,
            retrieval_mode="hybrid",
            trace_id="rag-trace-coverage",
            latency_ms=1.0,
        )


class ProductionNodeAdapterTests(unittest.TestCase):
    def test_gateway_plan_copies_only_explicit_case_and_authorizes_fixed_plan(self) -> None:
        state = AgentState(
            user_query="请为 Case-701 做路径规划",
            metadata={"model_gateway_metadata": {"mock_structured_output": planning_payload()}},
        )
        plan = GatewayRequestPlanner(
            MockQwenGateway(),
            request_metadata_provider=lambda item: item.metadata[
                "model_gateway_metadata"
            ],
        ).plan(state)

        self.assertEqual(TaskType.PLANNING_SAFETY, plan["task_type"])
        self.assertEqual("Case-701", plan["case_id"])
        self.assertEqual(
            [
                "generate_candidate_paths",
                "evaluate_path_safety",
                "evaluate_intraoperative_risk",
                "verify_skin_penetration",
            ],
            plan["tool_plan"],
        )
        self.assertEqual(["generate_candidate_paths"], plan["model_proposed_tools"])

    def test_gateway_rejects_invented_case_and_unauthorized_tool(self) -> None:
        for payload in (
            planning_payload(case_id="Case-999"),
            planning_payload(tool_plan=["run_segmentation"]),
        ):
            with self.subTest(payload=payload):
                state = AgentState(
                    user_query="请为 Case-701 做路径规划",
                    metadata={"model_gateway_metadata": {"mock_structured_output": payload}},
                )
                self.assertIsNone(
                    GatewayRequestPlanner(
                        MockQwenGateway(),
                        request_metadata_provider=lambda item: item.metadata[
                            "model_gateway_metadata"
                        ],
                    ).plan(state)
                )
                self.assertTrue(state.metadata["model_parse_failed"])
                self.assertEqual("MODEL_STRUCTURED_OUTPUT_INVALID", state.errors[-1]["code"])

    def test_production_handler_catalog_fails_safe_on_malformed_model_output(self) -> None:
        handlers = build_production_handlers(
            tool_executor=DeterministicMockToolExecutor(),
            model_gateway=MockQwenGateway(),
            rag_service=FakeRagService(),
            access_scope_provider=lambda _: ("public",),
            allow_test_controls=True,
        )
        state = AgentState(
            user_query="请为 Case-701 做路径规划",
            metadata={
                "model_gateway_metadata": {
                    "mock_structured_output": {"task_type": TaskType.PLANNING_SAFETY}
                }
            },
        )

        outcome = handlers["parse_request"](state, context("parse_request"))

        self.assertEqual(TaskType.UNKNOWN, state.task_type)
        self.assertEqual([], state.tool_plan)
        self.assertEqual("INVALID_MODEL_OUTPUT", outcome.output["status"])
        self.assertEqual([], state.tool_calls)

    def test_rag_adapter_keeps_compact_evidence_and_drops_chunk_text(self) -> None:
        service = FakeRagService()
        state = AgentState(
            user_query="请为 Case-701 做路径规划",
            task_type=TaskType.PLANNING_SAFETY,
            metadata={"access_scopes": ["public", "algorithm_team"]},
        )

        documents, citations = RagKnowledgeRetriever(
            service,
            access_scope_provider=lambda _: ("public", "algorithm_team"),
        ).retrieve(state)

        self.assertEqual(
            ("path_planning", "safety_evaluation"), service.requests[0].modules
        )
        self.assertEqual(("public", "algorithm_team"), service.requests[0].access_scopes)
        self.assertNotIn("text", documents[0])
        self.assertEqual("planning-rule", documents[0]["document_id"])
        self.assertEqual("planning-rule", citations[0]["document_id"])

    def test_rag_failure_is_sanitized_and_does_not_abort_routing(self) -> None:
        handlers = build_production_handlers(
            tool_executor=DeterministicMockToolExecutor(),
            model_gateway=MockQwenGateway(),
            rag_service=FailingRagService(),
            access_scope_provider=lambda _: ("public",),
        )
        state = AgentState(
            user_query="请为 Case-701 做路径规划",
            task_type=TaskType.PLANNING_SAFETY,
        )

        outcome = handlers["retrieve_project_knowledge"](
            state, context("retrieve_project_knowledge")
        )

        self.assertEqual([], state.retrieved_documents)
        self.assertEqual([], state.citations)
        self.assertFalse(state.metadata["rag_context_ready"])
        self.assertEqual("RAG_TIMEOUT", state.errors[-1]["code"])
        self.assertNotIn("private backend detail", str(state.errors))
        self.assertEqual("RAG_TIMEOUT", outcome.output["error_code"])

    def test_rag_gate_requires_complete_task_module_coverage(self) -> None:
        scenarios = (
            (
                TaskType.DATA_MODEL_VALIDATION,
                ("data_validation", "segmentation"),
                ("data_validation",),
                ["segmentation"],
            ),
            (
                TaskType.DATA_MODEL_VALIDATION,
                ("data_validation", "segmentation"),
                ("segmentation",),
                ["data_validation"],
            ),
            (
                TaskType.PLANNING_SAFETY,
                ("path_planning", "safety_evaluation"),
                ("path_planning",),
                ["safety_evaluation"],
            ),
            (
                TaskType.PLANNING_SAFETY,
                ("path_planning", "safety_evaluation"),
                ("safety_evaluation",),
                ["path_planning"],
            ),
        )
        for task_type, required_modules, available_modules, missing_modules in scenarios:
            with self.subTest(task_type=task_type):
                handlers = build_production_handlers(
                    tool_executor=DeterministicMockToolExecutor(),
                    model_gateway=MockQwenGateway(),
                    rag_service=ModuleCoverageRagService(available_modules),
                    access_scope_provider=lambda _: ("public",),
                )
                state = AgentState(user_query="test", task_type=task_type)

                outcome = handlers["retrieve_project_knowledge"](
                    state, context("retrieve_project_knowledge")
                )

                self.assertFalse(state.metadata["rag_context_ready"])
                self.assertEqual(
                    list(required_modules),
                    state.metadata["rag_required_modules"],
                )
                self.assertEqual(missing_modules, state.metadata["rag_missing_modules"])
                self.assertEqual("RAG_INCOMPLETE_EVIDENCE", state.errors[-1]["code"])
                self.assertFalse(state.errors[-1]["retryable"])
                self.assertEqual(
                    "RAG_INCOMPLETE_EVIDENCE", outcome.output["error_code"]
                )
                self.assertEqual(missing_modules, outcome.output["missing_modules"])

    def test_rag_gate_accepts_complete_task_module_coverage(self) -> None:
        scenarios = (
            (
                TaskType.DATA_MODEL_VALIDATION,
                ("data_validation", "segmentation"),
            ),
            (
                TaskType.PLANNING_SAFETY,
                ("path_planning", "safety_evaluation"),
            ),
        )
        for task_type, required_modules in scenarios:
            with self.subTest(task_type=task_type):
                handlers = build_production_handlers(
                    tool_executor=DeterministicMockToolExecutor(),
                    model_gateway=MockQwenGateway(),
                    rag_service=ModuleCoverageRagService(required_modules),
                    access_scope_provider=lambda _: ("public",),
                )
                state = AgentState(user_query="test", task_type=task_type)

                outcome = handlers["retrieve_project_knowledge"](
                    state, context("retrieve_project_knowledge")
                )

                self.assertTrue(state.metadata["rag_context_ready"])
                self.assertEqual([], state.metadata["rag_missing_modules"])
                self.assertEqual(list(required_modules), state.metadata["rag_evidence_modules"])
                self.assertNotIn("error_code", outcome.output)
                self.assertEqual(list(required_modules), outcome.output["evidence_modules"])

    def test_production_case_resolution_never_invents_mock_artifacts(self) -> None:
        handlers = build_production_handlers(
            tool_executor=DeterministicMockToolExecutor(),
            model_gateway=MockQwenGateway(),
            rag_service=FakeRagService(),
            access_scope_provider=lambda _: ("public",),
        )
        state = AgentState(
            user_query="请为 Case-701 做路径规划",
            case_id="Case-701",
            task_type=TaskType.PLANNING_SAFETY,
            metadata={"rag_context_ready": True},
        )

        outcome = handlers["resolve_case_context"](
            state, context("resolve_case_context")
        )

        self.assertEqual({}, state.artifacts)
        self.assertFalse(state.metadata["case_context_ready"])
        self.assertEqual(
            ["ct", "danger_masks", "skin", "skin_surface", "target"],
            outcome.output["missing_artifacts"],
        )

    def test_production_parse_rejects_test_only_fault_controls(self) -> None:
        handlers = build_production_handlers(
            tool_executor=DeterministicMockToolExecutor(),
            model_gateway=MockQwenGateway(),
            rag_service=FakeRagService(),
            access_scope_provider=lambda _: ("public",),
        )
        state = AgentState(
            user_query="请为 Case-701 做路径规划",
            metadata={"force_no_feasible_path": True},
        )

        outcome = handlers["parse_request"](state, context("parse_request"))

        self.assertEqual("INVALID_RUNTIME_METADATA", outcome.output["status"])
        self.assertEqual(TaskType.UNKNOWN, state.task_type)
        self.assertEqual("INVALID_ARGUMENT", state.errors[-1]["code"])


if __name__ == "__main__":
    unittest.main()
