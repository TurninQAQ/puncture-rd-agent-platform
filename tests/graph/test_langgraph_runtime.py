from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
import json
from math import ceil
import os
from statistics import median
import sys
import time
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent import (  # noqa: E402
    AgentState,
    AgentStatus,
    GraphExecutionError,
    TaskType,
    VerificationStatus,
    build_mock_handlers,
)
from puncture_agent.agent.langgraph_runtime import (  # noqa: E402
    LangGraphCheckpointError,
    LangGraphConcurrencyError,
    LangGraphDependencyError,
    LangGraphRuntime,
    langgraph_available,
)
from puncture_agent.agent.langgraph_state import RawBytesStateError  # noqa: E402
from puncture_agent.agent.production_nodes import build_production_handlers  # noqa: E402
from puncture_agent.agent.nodes import DeterministicMockToolExecutor  # noqa: E402
from puncture_agent.model_gateway.mock_qwen import MockQwenGateway  # noqa: E402
from puncture_agent.rag.mock_service import MockRagService  # noqa: E402


START = "__start__"
END = "__end__"


class FakeInMemorySaver:
    def __init__(self) -> None:
        self.values: dict[str, dict] = {}
        self._lock = Lock()
        self.put_count = 0

    def put(self, thread_id: str, value: dict) -> None:
        with self._lock:
            self.put_count += 1
            self.values[thread_id] = deepcopy(value)

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            value = self.values.get(thread_id)
            return deepcopy(value) if value is not None else None


@dataclass
class FakeSnapshot:
    values: dict
    tasks: tuple = ()


@dataclass
class FakeGraphOutput:
    value: dict | None
    interrupts: tuple = ()


class FakeStateGraph:
    def __init__(self, schema) -> None:
        self.schema = schema
        self.nodes = {}
        self.edges = {}
        self.conditional = {}

    def add_node(self, name, node) -> None:
        self.nodes[name] = node

    def add_edge(self, source, target) -> None:
        self.edges[source] = target

    def add_conditional_edges(self, source, route, path_map) -> None:
        self.conditional[source] = (route, path_map)

    def compile(self, checkpointer=None):
        return FakeCompiledGraph(self, checkpointer)


class FakeCompiledGraph:
    def __init__(self, builder: FakeStateGraph, checkpointer) -> None:
        self.builder = builder
        self.checkpointer = checkpointer
        self.durabilities = []

    def _thread_id(self, config) -> str | None:
        if not self.checkpointer:
            return None
        return config["configurable"]["thread_id"]

    def _next(self, current, state):
        if current in self.builder.conditional:
            route, path_map = self.builder.conditional[current]
            selected = route(state)
            return path_map[selected]
        return self.builder.edges[current]

    def _steps(self, input, config):
        state = deepcopy(input)
        current = START
        limit = config.get("recursion_limit", 100) if config else 100
        steps = 0
        while True:
            target = self._next(current, state)
            if target == END:
                break
            steps += 1
            if steps > limit:
                raise RuntimeError("fake recursion limit exceeded")
            node = self.builder.nodes[target]
            if isinstance(node, FakeCompiledGraph):
                update = node.invoke(state, config={"recursion_limit": limit})
            else:
                update = node(state)
            state.update(deepcopy(dict(update)))
            thread_id = self._thread_id(config)
            if thread_id:
                self.checkpointer.put(thread_id, state)
            current = target
            yield deepcopy(state)

    def invoke(self, input, config=None, durability=None, version="v1"):
        self.durabilities.append(durability)
        if input is None:
            thread_id = self._thread_id(config)
            value = self.checkpointer.get(thread_id) if thread_id else None
            return FakeGraphOutput(value) if version == "v2" else value
        final = deepcopy(input)
        for final in self._steps(input, config or {}):
            pass
        return FakeGraphOutput(final) if version == "v2" else final

    def stream(
        self,
        input,
        config=None,
        stream_mode=None,
        durability=None,
        subgraphs=False,
        version="v1",
    ):
        del stream_mode, subgraphs
        self.durabilities.append(durability)
        def encode(value):
            if version == "v2":
                return {
                    "type": "values",
                    "ns": (),
                    "data": value,
                    "interrupts": (),
                }
            return value

        yield encode(deepcopy(input))
        for value in self._steps(input, config or {}):
            yield encode(value)

    def get_state(self, config, subgraphs=False):
        del subgraphs
        thread_id = self._thread_id(config)
        return FakeSnapshot(self.checkpointer.get(thread_id) or {})

    def update_state(self, config, values, as_node=None):
        del as_node
        thread_id = self._thread_id(config)
        if thread_id and values is not None:
            self.checkpointer.put(thread_id, dict(values))
        return config


class FakeLangGraphApi:
    class GraphBubbleUp(Exception):
        pass

    class GraphRecursionError(Exception):
        pass

    class Command:
        def __init__(self, *, resume):
            self.resume = resume

    state_graph = FakeStateGraph
    start = START
    end = END
    in_memory_saver = FakeInMemorySaver
    graph_bubble_up = GraphBubbleUp
    graph_recursion_error = GraphRecursionError
    command = Command


class TransportFaultExecutor:
    def __init__(
        self,
        *,
        fail_once: tuple[str, ...] = (),
        fail_always: tuple[str, ...] = (),
        no_candidate: bool = False,
    ) -> None:
        self.delegate = DeterministicMockToolExecutor()
        self.fail_once = set(fail_once)
        self.fail_always = set(fail_always)
        self.failed_once: set[str] = set()
        self.no_candidate = no_candidate
        self.calls: list[str] = []

    def execute(self, tool_name, request):
        self.calls.append(tool_name)
        if tool_name in self.fail_always:
            raise TimeoutError("private timeout detail")
        if tool_name in self.fail_once and tool_name not in self.failed_once:
            self.failed_once.add(tool_name)
            raise TimeoutError("private timeout detail")
        if self.no_candidate and tool_name == "generate_candidate_paths":
            return {
                "status": "FAILED",
                "result": None,
                "error": {
                    "code": "NO_CANDIDATE_PATH",
                    "message": "no valid candidate satisfies the constraints",
                    "retryable": False,
                },
            }
        return self.delegate.execute(tool_name, request)


def build_runtime() -> LangGraphRuntime:
    return LangGraphRuntime(
        PROJECT_ROOT / "graph" / "main_graph.json",
        build_mock_handlers(),
        langgraph_api=FakeLangGraphApi(),
    )


class ProductionLangGraphRuntimeTests(unittest.TestCase):
    def test_planning_success_matches_reference_path_and_uses_sync_durability(self) -> None:
        runtime = build_runtime()
        state = runtime.run(
            AgentState(
                user_query="请对 Case-102 做路径规划和皮肤穿透安全评估",
                session_id="langgraph-plan-success",
            )
        )

        self.assertEqual(AgentStatus.SUCCEEDED, state.status)
        self.assertEqual(VerificationStatus.PASS, state.verification_status)
        self.assertIn("planning_safety_subgraph", state.visited_nodes)
        self.assertIn(
            "planning_safety_subgraph.generate_candidate_paths", state.visited_nodes
        )
        self.assertEqual(
            [
                "generate_candidate_paths",
                "evaluate_path_safety",
                "evaluate_intraoperative_risk",
                "verify_skin_penetration",
            ],
            [call["tool_name"] for call in state.tool_calls],
        )
        self.assertEqual("sync", runtime.compiled_graph.durabilities[-1])

    def test_mcs_data_flow_and_validation_stop_branches(self) -> None:
        runtime = build_runtime()
        success = runtime.run(
            AgentState(
                user_query="检查 Case-203 的 MCS 标签和分割模型结果",
                session_id="langgraph-data-success",
            )
        )
        self.assertEqual(AgentStatus.SUCCEEDED, success.status)
        self.assertEqual(
            [
                "inspect_case_metadata",
                "convert_mcs_to_nifti",
                "validate_label_schema",
                "run_segmentation",
                "validate_segmentation_result",
                "extract_skin_surface",
            ],
            [call["tool_name"] for call in success.tool_calls],
        )

        failed = runtime.run(
            AgentState(
                user_query="检查 Case-204 的标签和分割",
                session_id="langgraph-data-geometry-failure",
                metadata={"force_geometry_mismatch": True},
            )
        )
        self.assertEqual(AgentStatus.MANUAL_REVIEW, failed.status)
        self.assertEqual(
            ["inspect_case_metadata"],
            [call["tool_name"] for call in failed.tool_calls],
        )

        label_failed = runtime.run(
            AgentState(
                user_query="检查 Case-205 的 MCS 标签和分割",
                session_id="langgraph-data-label-failure",
                metadata={"force_label_schema_error": True},
            )
        )
        self.assertEqual(AgentStatus.MANUAL_REVIEW, label_failed.status)
        self.assertEqual(
            [
                "inspect_case_metadata",
                "convert_mcs_to_nifti",
                "validate_label_schema",
            ],
            [call["tool_name"] for call in label_failed.tool_calls],
        )

    def test_missing_case_and_missing_artifact_never_call_algorithm_tools(self) -> None:
        runtime = build_runtime()
        missing_case = runtime.run(
            AgentState(
                user_query="执行路径规划和安全评估",
                session_id="langgraph-missing-case",
            )
        )
        self.assertEqual(AgentStatus.AWAITING_INPUT, missing_case.status)
        self.assertEqual([], missing_case.tool_calls)

        missing_target = runtime.run(
            AgentState(
                user_query="对 Case-408 做路径规划",
                session_id="langgraph-missing-target",
                metadata={"missing_required_artifacts": ["target"]},
            )
        )
        self.assertEqual(AgentStatus.AWAITING_INPUT, missing_target.status)
        self.assertEqual([], missing_target.tool_calls)

    def test_no_feasible_path_is_a_valid_terminal_outcome(self) -> None:
        state = build_runtime().run(
            AgentState(
                user_query="为 Case-304 进行针道规划",
                session_id="langgraph-no-path",
                metadata={"force_no_feasible_path": True},
            )
        )
        self.assertEqual(AgentStatus.COMPLETED_WITH_NO_RESULT, state.status)
        self.assertEqual(VerificationStatus.NO_FEASIBLE_PATH, state.verification_status)
        self.assertNotIn(
            "evaluate_path_safety", [call["tool_name"] for call in state.tool_calls]
        )

    def test_frozen_no_candidate_error_is_a_valid_terminal_outcome(self) -> None:
        executor = TransportFaultExecutor(no_candidate=True)
        runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(executor),
            langgraph_api=FakeLangGraphApi(),
        )

        state = runtime.run(
            AgentState(
                user_query="为 Case-305 进行针道规划",
                session_id="langgraph-frozen-no-candidate",
            )
        )

        self.assertEqual(AgentStatus.COMPLETED_WITH_NO_RESULT, state.status)
        self.assertEqual(VerificationStatus.NO_FEASIBLE_PATH, state.verification_status)
        self.assertEqual(["generate_candidate_paths"], executor.calls)
        self.assertEqual("NO_RESULT", state.tool_calls[0]["status"])

    def test_retry_policy_records_exact_calls(self) -> None:
        runtime = build_runtime()
        recovered = runtime.run(
            AgentState(
                user_query="对 Case-405 做路径规划",
                session_id="langgraph-retry-once",
                metadata={"fail_tool_once": ["generate_candidate_paths"]},
            )
        )
        calls = [
            call
            for call in recovered.tool_calls
            if call["tool_name"] == "generate_candidate_paths"
        ]
        self.assertEqual(["FAILED", "SUCCESS"], [call["status"] for call in calls])
        self.assertEqual(1, recovered.retry_count)

        exhausted = runtime.run(
            AgentState(
                user_query="对 Case-406 做路径规划",
                session_id="langgraph-retry-exhausted",
                metadata={"fail_tool_always": ["generate_candidate_paths"]},
            )
        )
        persistent_calls = [
            call
            for call in exhausted.tool_calls
            if call["tool_name"] == "generate_candidate_paths"
        ]
        self.assertEqual(2, len(persistent_calls))
        self.assertEqual(AgentStatus.MANUAL_REVIEW, exhausted.status)

        denied = runtime.run(
            AgentState(
                user_query="对 Case-407 做路径规划",
                session_id="langgraph-no-retry",
                metadata={"fail_tool_non_retryable": ["generate_candidate_paths"]},
            )
        )
        denied_calls = [
            call
            for call in denied.tool_calls
            if call["tool_name"] == "generate_candidate_paths"
        ]
        self.assertEqual(1, len(denied_calls))
        self.assertEqual(0, denied.retry_count)

    def test_transport_timeouts_are_retryable_and_block_downstream_tools(self) -> None:
        segmentation_executor = TransportFaultExecutor(
            fail_once=("run_segmentation",)
        )
        segmentation_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(segmentation_executor),
            langgraph_api=FakeLangGraphApi(),
        )
        recovered = segmentation_runtime.run(
            AgentState(
                user_query="检查 Case-409 的 MCS 标签和分割",
                session_id="langgraph-transport-timeout-recovery",
            )
        )
        self.assertEqual(AgentStatus.SUCCEEDED, recovered.status)
        self.assertEqual(2, segmentation_executor.calls.count("run_segmentation"))
        self.assertEqual(
            1,
            segmentation_executor.calls.count("validate_segmentation_result"),
        )
        timeout_error = next(
            item for item in recovered.errors if item["code"] == "TIMEOUT"
        )
        self.assertTrue(timeout_error["retryable"])
        self.assertNotIn("private timeout detail", str(recovered.errors))

        risk_executor = TransportFaultExecutor(
            fail_always=("evaluate_intraoperative_risk",)
        )
        risk_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(risk_executor),
            langgraph_api=FakeLangGraphApi(),
        )
        exhausted = risk_runtime.run(
            AgentState(
                user_query="对 Case-410 做路径规划",
                session_id="langgraph-risk-timeout-exhausted",
            )
        )
        self.assertEqual(AgentStatus.MANUAL_REVIEW, exhausted.status)
        self.assertEqual(2, risk_executor.calls.count("evaluate_intraoperative_risk"))
        self.assertEqual(0, risk_executor.calls.count("verify_skin_penetration"))

        class StringRetryExecutor:
            def __init__(self):
                self.calls = 0

            def execute(self, tool_name, request):
                del tool_name, request
                self.calls += 1
                return {
                    "status": "FAILED",
                    "result": None,
                    "error": {
                        "code": "DEPENDENCY_FAILED",
                        "message": "malformed retry flag",
                        "retryable": "false",
                    },
                }

        malformed_executor = StringRetryExecutor()
        malformed_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(malformed_executor),
            langgraph_api=FakeLangGraphApi(),
        )
        malformed = malformed_runtime.run(
            AgentState(
                user_query="对 Case-411 做路径规划",
                session_id="langgraph-malformed-retry-flag",
            )
        )
        self.assertEqual(AgentStatus.MANUAL_REVIEW, malformed.status)
        self.assertEqual(0, malformed.retry_count)
        self.assertEqual(1, malformed_executor.calls)

    def test_checkpoint_and_stream_are_thread_isolated(self) -> None:
        runtime = build_runtime()
        streamed = AgentState(
            user_query="对 Case-501 做路径规划",
            session_id="langgraph-stream",
        )
        event_iterator = runtime.stream(streamed)
        first_event = next(event_iterator)
        # The API-facing iterator buffers transitions until the synchronous
        # stream has drained, so a visible event cannot precede its checkpoint.
        durable_before_ack = runtime.checkpoint_state(thread_id=streamed.session_id)
        self.assertEqual(streamed.to_dict(), durable_before_ack.to_dict())
        resumed_while_events_are_buffered = runtime.resume(
            thread_id=streamed.session_id
        )
        self.assertEqual(
            streamed.to_dict(), resumed_while_events_are_buffered.to_dict()
        )
        events = [first_event, *event_iterator]
        self.assertEqual("RUN_COMPLETED", events[-1].event_type)
        self.assertEqual(
            list(range(1, len(events) + 1)), [event.sequence for event in events]
        )
        self.assertEqual("sync", runtime.compiled_graph.durabilities[-1])
        checkpoint = runtime.checkpoint_state(thread_id=streamed.session_id)
        self.assertEqual(streamed.to_dict(), checkpoint.to_dict())

        def execute(index: int):
            case = f"Case-{600 + index}"
            return runtime.run(
                AgentState(
                    user_query=f"对 {case} 做路径规划",
                    session_id=f"isolated-{index}",
                )
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            states = list(pool.map(execute, range(20)))
        self.assertEqual(20, len({state.session_id for state in states}))
        for index, state in enumerate(states):
            self.assertEqual(f"Case-{600 + index}", state.case_id)
            stored = runtime.checkpoint_state(thread_id=state.session_id)
            self.assertEqual(state.case_id, stored.case_id)

        entered = Event()
        release = Event()
        handlers = dict(build_mock_handlers())
        original_parse = handlers["parse_request"]

        def blocking_parse(state, context):
            entered.set()
            if not release.wait(timeout=2.0):
                raise RuntimeError("test synchronization timeout")
            return original_parse(state, context)

        handlers["parse_request"] = blocking_parse
        single_flight_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            handlers,
            langgraph_api=FakeLangGraphApi(),
        )
        first_state = AgentState(
            user_query="对 Case-699 做路径规划",
            session_id="same-thread-single-flight",
        )
        second_state = AgentState(
            user_query="对 Case-699 做路径规划",
            session_id="same-thread-single-flight",
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(single_flight_runtime.run, first_state)
            self.assertTrue(entered.wait(timeout=2.0))
            second_future = pool.submit(single_flight_runtime.run, second_state)
            with self.assertRaises(LangGraphConcurrencyError):
                second_future.result(timeout=2.0)
            release.set()
            first_result = first_future.result(timeout=2.0)
        self.assertEqual(AgentStatus.SUCCEEDED, first_result.status)
        self.assertEqual(4, len(first_result.tool_calls))
        self.assertEqual(AgentStatus.CREATED, second_state.status)

    def test_missing_input_can_restart_from_start_with_same_thread(self) -> None:
        runtime = build_runtime()
        waiting = runtime.run(
            AgentState(
                user_query="执行路径规划和安全评估",
                session_id="langgraph-missing-input-resume",
            )
        )
        self.assertEqual(AgentStatus.AWAITING_INPUT, waiting.status)

        resumed = runtime.resume(
            thread_id=waiting.session_id,
            updates={"case_id": "Case-777"},
        )

        self.assertEqual(AgentStatus.SUCCEEDED, resumed.status)
        self.assertEqual("Case-777", resumed.case_id)
        self.assertEqual(2, resumed.visited_nodes.count("parse_request"))
        self.assertEqual(
            resumed.to_dict(),
            runtime.checkpoint_state(thread_id=waiting.session_id).to_dict(),
        )
        with self.assertRaisesRegex(ValueError, "AWAITING_INPUT"):
            runtime.resume(
                thread_id=waiting.session_id,
                updates={"case_id": "Case-778"},
            )

    def test_thread_id_must_match_state_and_postgres_limit(self) -> None:
        runtime = build_runtime()
        state = AgentState(user_query="test", session_id="thread-a")
        with self.assertRaisesRegex(ValueError, "must match"):
            runtime.run(state, thread_id="thread-b")
        state.session_id = "x" * 255
        with self.assertRaisesRegex(ValueError, "fewer than 255"):
            runtime.run(state)
        reserved = AgentState(
            user_query="test",
            session_id="reserved-runtime-metadata",
            metadata={"_langgraph_subgraph_steps": "bad"},
        )
        with self.assertRaisesRegex(ValueError, "reserved"):
            runtime.run(reserved)
        invalid = AgentState(
            user_query="test",
            session_id="invalid-initial-state",
            metadata={"raw": b"forbidden"},
        )
        with self.assertRaises(RawBytesStateError):
            runtime.run(invalid)
        self.assertEqual(AgentStatus.CREATED, invalid.status)
        self.assertNotIn("trace_id", invalid.metadata)

    def test_production_model_rag_and_tool_adapters_compose(self) -> None:
        handlers = build_production_handlers(
            tool_executor=DeterministicMockToolExecutor(),
            model_gateway=MockQwenGateway(),
            rag_service=MockRagService.from_default_fixture(),
            access_scope_provider=lambda _: ("public", "algorithm_team"),
            allow_test_controls=True,
        )
        runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            handlers,
            langgraph_api=FakeLangGraphApi(),
        )
        state = AgentState(
            user_query="planning safety for Case-710 with needle constraints",
            session_id="production-adapters-success",
            artifacts={
                "ct": "artifact-Case-710-ct",
                "skin": "artifact-Case-710-skin",
                "skin_surface": "artifact-Case-710-skin-surface",
                "target": "artifact-Case-710-target",
                "danger_masks": {
                    "heart": "artifact-Case-710-heart",
                    "bone": "artifact-Case-710-bone",
                    "bronchus": "artifact-Case-710-bronchus",
                    "vessel": "artifact-Case-710-vessel",
                    "lung": "artifact-Case-710-lung",
                },
            },
            metadata={
                "access_scopes": ["public", "algorithm_team"],
                "model_gateway_metadata": {
                    "mock_structured_output": {
                        "task_type": TaskType.PLANNING_SAFETY,
                        "case_id": "Case-710",
                        "tool_plan": ["generate_candidate_paths"],
                        "input_format": "AUTO",
                        "run_segmentation": False,
                        "extract_skin_surface": False,
                    }
                },
            },
        )

        runtime.run(state)

        self.assertEqual(AgentStatus.SUCCEEDED, state.status)
        self.assertTrue(state.retrieved_documents)
        self.assertEqual(4, len(state.tool_calls))
        self.assertNotIn("use_mock_artifacts", state.metadata)

    def test_malformed_model_output_fails_before_any_tool_execution(self) -> None:
        executor = DeterministicMockToolExecutor()
        handlers = build_production_handlers(
            tool_executor=executor,
            model_gateway=MockQwenGateway(),
            rag_service=MockRagService.from_default_fixture(),
            access_scope_provider=lambda _: ("public",),
            allow_test_controls=True,
        )
        runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            handlers,
            langgraph_api=FakeLangGraphApi(),
        )
        state = AgentState(
            user_query="请为 Case-711 做路径规划",
            session_id="production-malformed-model",
            metadata={
                "model_gateway_metadata": {
                    "mock_structured_output": {"task_type": TaskType.PLANNING_SAFETY}
                }
            },
        )

        runtime.run(state)

        self.assertEqual(AgentStatus.AWAITING_INPUT, state.status)
        self.assertEqual([], state.tool_calls)
        self.assertEqual("MODEL_STRUCTURED_OUTPUT_INVALID", state.errors[0]["code"])

    def test_production_label_failure_stops_before_segmentation(self) -> None:
        executor = DeterministicMockToolExecutor()
        handlers = build_production_handlers(
            tool_executor=executor,
            model_gateway=MockQwenGateway(),
            rag_service=MockRagService.from_default_fixture(),
            access_scope_provider=lambda _: ("public", "algorithm_team"),
            allow_test_controls=True,
        )
        runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            handlers,
            langgraph_api=FakeLangGraphApi(),
        )
        state = AgentState(
            user_query="validate MCS label schema and segmentation for Case-712",
            session_id="production-label-schema-failure",
            artifacts={
                "ct": "artifact-Case-712-ct",
                "raw_labels": "artifact-Case-712-raw-labels",
            },
            metadata={
                "force_label_schema_error": True,
                "model_gateway_metadata": {
                    "mock_structured_output": {
                        "task_type": TaskType.DATA_MODEL_VALIDATION,
                        "case_id": "Case-712",
                        "tool_plan": ["validate_label_schema"],
                        "input_format": "MCS",
                        "run_segmentation": True,
                        "extract_skin_surface": True,
                    }
                },
            },
        )

        runtime.run(state)

        self.assertEqual(AgentStatus.MANUAL_REVIEW, state.status)
        self.assertEqual(
            [
                "inspect_case_metadata",
                "convert_mcs_to_nifti",
                "validate_label_schema",
            ],
            [call["tool_name"] for call in state.tool_calls],
        )

    def test_missing_optional_dependency_is_explicit(self) -> None:
        if langgraph_available():
            self.skipTest("real LangGraph is installed in this environment")
        with self.assertRaises(LangGraphDependencyError):
            LangGraphRuntime(
                PROJECT_ROOT / "graph" / "main_graph.json",
                build_mock_handlers(),
            )

    def test_fake_dependency_orchestration_p95_and_checkpoint_size_gate(self) -> None:
        runtime = build_runtime()
        for index in range(5):
            runtime.run(
                AgentState(
                    user_query=f"对 Case-{820 + index} 做路径规划",
                    session_id=f"benchmark-warmup-{index}",
                )
            )

        durations_ms = []
        final = None
        for index in range(50):
            started = time.perf_counter()
            final = runtime.run(
                AgentState(
                    user_query=f"对 Case-{900 + index} 做路径规划",
                    session_id=f"benchmark-sample-{index}",
                )
            )
            durations_ms.append((time.perf_counter() - started) * 1000.0)

        p95 = sorted(durations_ms)[ceil(len(durations_ms) * 0.95) - 1]
        self.assertLessEqual(p95, 100.0, f"graph-only P95 was {p95:.3f} ms")
        encoded = json.dumps(final.to_dict(), ensure_ascii=False).encode("utf-8")
        self.assertLessEqual(len(encoded), 1024 * 1024)


@unittest.skipUnless(langgraph_available(), "real LangGraph dependency is not installed")
class RealLangGraphSmokeTests(unittest.TestCase):
    def test_real_stategraph_planning_and_checkpoint(self) -> None:
        runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(),
        )
        state = runtime.run(
            AgentState(
                user_query="对 Case-801 做路径规划",
                session_id="real-langgraph-smoke",
            )
        )
        self.assertEqual(AgentStatus.SUCCEEDED, state.status)
        self.assertEqual(
            state.to_dict(),
            runtime.checkpoint_state(thread_id=state.session_id).to_dict(),
        )

    def test_real_stategraph_graph_only_p95_records_engineering_gate(self) -> None:
        runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(),
        )
        for index in range(5):
            runtime.run(
                AgentState(
                    user_query=f"对 Case-{850 + index} 做路径规划",
                    session_id=f"real-benchmark-warmup-{index}",
                )
            )
        round_p95_ms = []
        all_durations_ms = []
        final_state = None
        for round_index in range(3):
            durations_ms = []
            for index in range(50):
                started = time.perf_counter()
                final_state = runtime.run(
                    AgentState(
                        user_query=f"对 Case-{950 + index} 做路径规划",
                        session_id=(
                            f"real-benchmark-{round_index}-sample-{index}"
                        ),
                    )
                )
                durations_ms.append((time.perf_counter() - started) * 1000.0)
            all_durations_ms.extend(durations_ms)
            round_p95_ms.append(
                sorted(durations_ms)[ceil(len(durations_ms) * 0.95) - 1]
            )
        p95 = median(round_p95_ms)
        ordered_durations = sorted(all_durations_ms)
        aggregate_p50 = median(ordered_durations)
        aggregate_p95 = ordered_durations[
            ceil(len(ordered_durations) * 0.95) - 1
        ]
        checkpoint_bytes = len(
            json.dumps(final_state.to_dict(), ensure_ascii=False).encode("utf-8")
        )
        print(
            "LANGGRAPH_BENCHMARK "
            f"round_p95_ms={','.join(f'{value:.3f}' for value in round_p95_ms)} "
            f"median_round_p95_ms={p95:.3f} "
            f"aggregate_p50_ms={aggregate_p50:.3f} "
            f"aggregate_p95_ms={aggregate_p95:.3f} "
            f"max_ms={max(ordered_durations):.3f} "
            f"checkpoint_bytes={checkpoint_bytes}"
        )
        self.assertGreater(p95, 0.0)
        self.assertLess(p95, 1000.0, f"real LangGraph P95 was {p95:.3f} ms")
        self.assertLessEqual(checkpoint_bytes, 1024 * 1024)
        if os.environ.get("PUNCTURE_ENFORCE_PERFORMANCE_GATES") == "1":
            self.assertLessEqual(
                p95,
                100.0,
                f"real LangGraph P95 was {p95:.3f} ms",
            )

    def test_real_interrupt_stream_resume_reuses_trace(self) -> None:
        from langgraph.types import interrupt
        from puncture_agent.observability.tracing import (
            InMemoryTraceExporter,
            TraceRecorder,
        )

        handlers = dict(build_mock_handlers())
        original = handlers["parse_request"]

        def approval_gate(state, context):
            decision = interrupt(
                {"kind": "operator_approval", "prompt": "approve workflow"}
            )
            state.metadata["operator_decision"] = decision
            return original(state, context)

        handlers["parse_request"] = approval_gate
        exporter = InMemoryTraceExporter()
        runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            handlers,
            tracer=TraceRecorder(exporter),
        )
        state = AgentState(
            user_query="对 Case-802 做路径规划",
            session_id="real-langgraph-interrupt",
        )

        events = list(runtime.stream(state))

        self.assertEqual("RUN_INTERRUPTED", events[-1].event_type)
        self.assertEqual(AgentStatus.AWAITING_INPUT, state.status)
        self.assertEqual("operator_approval", state.metadata["pending_interrupts"][0]["value"]["kind"])
        trace_id = state.metadata["trace_id"]
        checkpoint = runtime.checkpoint_state(thread_id=state.session_id)
        self.assertEqual(AgentStatus.AWAITING_INPUT, checkpoint.status)
        self.assertEqual(
            state.metadata["pending_interrupts"],
            checkpoint.metadata["pending_interrupts"],
        )

        resumed = runtime.resume(
            thread_id=state.session_id,
            resume_value={"approved": True},
        )

        self.assertEqual(AgentStatus.SUCCEEDED, resumed.status)
        self.assertEqual({"approved": True}, resumed.metadata["operator_decision"])
        self.assertEqual(trace_id, resumed.metadata["trace_id"])
        spans = exporter.spans()
        self.assertEqual({trace_id}, {span.trace_id for span in spans})
        self.assertEqual(2, len([span for span in spans if span.name == "agent.graph"]))
        self.assertTrue(all(span.status == "OK" for span in spans))
        interrupted_spans = [
            span
            for span in spans
            if any(event["name"] == "agent.node.interrupted" for event in span.events)
        ]
        self.assertEqual(1, len(interrupted_spans))

    def test_real_new_runtime_resumes_child_from_safe_checkpoint(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import interrupt

        executor = TransportFaultExecutor()
        handlers = dict(build_mock_handlers(executor))
        original_router = handlers["candidate_router"]

        def approval_gate(state, context):
            decision = interrupt(
                {"kind": "candidate_review", "prompt": "approve candidates"}
            )
            state.metadata["candidate_review"] = decision
            return original_router(state, context)

        handlers["candidate_router"] = approval_gate
        saver = InMemorySaver()
        session_id = "real-langgraph-safe-checkpoint"
        first_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            handlers,
            checkpointer=saver,
        )
        initial = AgentState(
            user_query="对 Case-803 做路径规划",
            session_id=session_id,
        )
        events = list(first_runtime.stream(initial))
        interrupted = initial

        self.assertEqual(AgentStatus.AWAITING_INPUT, interrupted.status)
        self.assertEqual("RUN_INTERRUPTED", events[-1].event_type)
        self.assertNotIn(
            "planning_safety_subgraph",
            [event.node_id for event in events if event.event_type == "NODE_COMPLETED"],
        )
        self.assertEqual(["generate_candidate_paths"], executor.calls)

        second_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            handlers,
            checkpointer=saver,
        )
        restored = second_runtime.checkpoint_state(thread_id=session_id)
        self.assertEqual(AgentStatus.AWAITING_INPUT, restored.status)
        self.assertEqual(
            "candidate_review",
            restored.metadata["pending_interrupts"][0]["value"]["kind"],
        )
        self.assertEqual(
            ["generate_candidate_paths"],
            [call["tool_name"] for call in restored.tool_calls],
        )
        resumed = second_runtime.resume(
            thread_id=session_id,
            resume_value={"approved": True},
        )

        self.assertEqual(AgentStatus.SUCCEEDED, resumed.status)
        self.assertEqual(
            [
                "generate_candidate_paths",
                "evaluate_path_safety",
                "evaluate_intraoperative_risk",
                "verify_skin_penetration",
            ],
            executor.calls,
        )
        self.assertEqual(
            1,
            resumed.visited_nodes.count(
                "planning_safety_subgraph.generate_candidate_paths"
            ),
        )

    def test_real_unexpected_child_failure_is_durable_and_terminal(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        executor = TransportFaultExecutor()
        handlers = dict(build_mock_handlers(executor))
        failure_count = 0

        def fail_safety(state, context):
            nonlocal failure_count
            del state, context
            failure_count += 1
            raise RuntimeError("private handler detail")

        handlers["evaluate_path_safety"] = fail_safety
        saver = InMemorySaver()
        session_id = "real-langgraph-durable-failure"
        state = AgentState(
            user_query="对 Case-804 做路径规划",
            session_id=session_id,
        )
        first_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            handlers,
            checkpointer=saver,
        )

        with self.assertRaises(GraphExecutionError):
            first_runtime.run(state)

        self.assertEqual(AgentStatus.FAILED, state.status)
        self.assertEqual("UNHANDLED_NODE_ERROR", state.errors[-1]["code"])
        self.assertNotIn("private handler detail", str(state.errors))
        self.assertEqual(["generate_candidate_paths"], executor.calls)
        self.assertEqual(1, failure_count)

        second_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            handlers,
            checkpointer=saver,
        )
        restored = second_runtime.checkpoint_state(thread_id=session_id)
        self.assertEqual(AgentStatus.FAILED, restored.status)
        self.assertEqual(state.to_dict(), restored.to_dict())

        resumed = second_runtime.resume(thread_id=session_id)
        self.assertEqual(AgentStatus.FAILED, resumed.status)
        self.assertEqual(1, failure_count)
        self.assertEqual(["generate_candidate_paths"], executor.calls)

        recursion_executor = TransportFaultExecutor()
        recursion_saver = InMemorySaver()
        recursion_session = "real-langgraph-recursion-terminal"
        recursion_state = AgentState(
            user_query="对 Case-805 做路径规划",
            session_id=recursion_session,
        )
        recursion_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(recursion_executor),
            checkpointer=recursion_saver,
        )
        with self.assertRaisesRegex(GraphExecutionError, "recursion_limit=5"):
            recursion_runtime.run(
                recursion_state,
                config={"recursion_limit": 5},
            )
        self.assertEqual(AgentStatus.FAILED, recursion_state.status)
        self.assertEqual(
            5,
            recursion_state.errors[-1]["details"]["requested_recursion_limit"],
        )
        calls_after_limit = list(recursion_executor.calls)
        self.assertTrue(calls_after_limit)
        recursion_checkpoint = recursion_runtime.checkpoint_state(
            thread_id=recursion_session
        )
        self.assertEqual(recursion_state.to_dict(), recursion_checkpoint.to_dict())
        recursion_resumed = recursion_runtime.resume(thread_id=recursion_session)
        self.assertEqual(AgentStatus.FAILED, recursion_resumed.status)
        self.assertEqual(calls_after_limit, recursion_executor.calls)

        class OneShotFailingSaver(InMemorySaver):
            def __init__(self):
                super().__init__()
                self.failed = False

            def put(self, config, checkpoint, metadata, new_versions):
                stored = super().put(config, checkpoint, metadata, new_versions)
                values = checkpoint.get("channel_values", {})
                if (
                    not self.failed
                    and len(values.get("tool_calls", ())) == 1
                    and str(values.get("current_node", "")).endswith(
                        ".generate_candidate_paths"
                    )
                ):
                    self.failed = True
                    raise OSError("private checkpoint outage")
                return stored

        uncertain_executor = TransportFaultExecutor()
        uncertain_saver = OneShotFailingSaver()
        uncertain_session = "real-langgraph-checkpoint-uncertain"
        uncertain_state = AgentState(
            user_query="对 Case-806 做路径规划",
            session_id=uncertain_session,
        )
        uncertain_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(uncertain_executor),
            checkpointer=uncertain_saver,
        )
        with self.assertRaises(LangGraphCheckpointError):
            uncertain_runtime.run(uncertain_state)
        self.assertEqual(AgentStatus.MANUAL_REVIEW, uncertain_state.status)
        self.assertTrue(uncertain_state.metadata["execution_state_uncertain"])
        self.assertEqual(
            "CHECKPOINT_DURABILITY_UNCERTAIN",
            uncertain_state.errors[-1]["code"],
        )
        self.assertEqual(["generate_candidate_paths"], uncertain_executor.calls)
        uncertain_checkpoint = uncertain_runtime.checkpoint_state(
            thread_id=uncertain_session
        )
        self.assertEqual(uncertain_state.to_dict(), uncertain_checkpoint.to_dict())
        with self.assertRaisesRegex(
            LangGraphCheckpointError,
            "manual checkpoint reconciliation",
        ):
            uncertain_runtime.resume(thread_id=uncertain_session)
        reconciled_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(uncertain_executor),
            checkpointer=uncertain_saver,
        )
        uncertain_resumed = reconciled_runtime.resume(thread_id=uncertain_session)
        self.assertEqual(AgentStatus.MANUAL_REVIEW, uncertain_resumed.status)
        self.assertEqual(["generate_candidate_paths"], uncertain_executor.calls)

        class FailingTerminalUpdateSaver(InMemorySaver):
            def put(self, config, checkpoint, metadata, new_versions):
                stored = super().put(config, checkpoint, metadata, new_versions)
                if metadata.get("source") == "update":
                    raise OSError("terminal update unavailable")
                return stored

        combined_saver = FailingTerminalUpdateSaver()
        combined_handlers = dict(build_mock_handlers())
        combined_failures = 0

        def combined_handler_failure(state, context):
            nonlocal combined_failures
            del state, context
            combined_failures += 1
            raise RuntimeError("private combined failure")

        combined_handlers["parse_request"] = combined_handler_failure
        combined_session = "real-langgraph-failure-update-uncertain"
        combined_state = AgentState(
            user_query="对 Case-808 做路径规划",
            session_id=combined_session,
        )
        combined_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            combined_handlers,
            checkpointer=combined_saver,
        )
        with self.assertRaises(LangGraphCheckpointError):
            combined_runtime.run(combined_state)
        self.assertEqual(AgentStatus.MANUAL_REVIEW, combined_state.status)
        self.assertTrue(combined_state.metadata["execution_state_uncertain"])
        self.assertEqual(1, combined_failures)
        with self.assertRaises(LangGraphCheckpointError):
            combined_runtime.resume(thread_id=combined_session)
        combined_restarted = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            combined_handlers,
            checkpointer=combined_saver,
        )
        combined_checkpoint = combined_restarted.checkpoint_state(
            thread_id=combined_session
        )
        self.assertEqual(AgentStatus.MANUAL_REVIEW, combined_checkpoint.status)
        combined_resumed = combined_restarted.resume(thread_id=combined_session)
        self.assertEqual(AgentStatus.MANUAL_REVIEW, combined_resumed.status)
        self.assertEqual(1, combined_failures)

        recursion_update_executor = TransportFaultExecutor()
        recursion_update_saver = FailingTerminalUpdateSaver()
        recursion_update_session = "real-langgraph-recursion-update-uncertain"
        recursion_update_state = AgentState(
            user_query="对 Case-809 做路径规划",
            session_id=recursion_update_session,
        )
        recursion_update_runtime = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(recursion_update_executor),
            checkpointer=recursion_update_saver,
        )
        with self.assertRaises(LangGraphCheckpointError):
            recursion_update_runtime.run(
                recursion_update_state,
                config={"recursion_limit": 5},
            )
        self.assertEqual(AgentStatus.MANUAL_REVIEW, recursion_update_state.status)
        self.assertTrue(recursion_update_state.metadata["execution_state_uncertain"])
        recursion_update_calls = list(recursion_update_executor.calls)
        with self.assertRaises(LangGraphCheckpointError):
            recursion_update_runtime.resume(thread_id=recursion_update_session)
        recursion_update_restarted = LangGraphRuntime(
            PROJECT_ROOT / "graph" / "main_graph.json",
            build_mock_handlers(recursion_update_executor),
            checkpointer=recursion_update_saver,
        )
        recursion_update_resumed = recursion_update_restarted.resume(
            thread_id=recursion_update_session
        )
        self.assertEqual(AgentStatus.MANUAL_REVIEW, recursion_update_resumed.status)
        self.assertEqual(recursion_update_calls, recursion_update_executor.calls)

    def test_real_contract_and_state_boundary_failures_are_durable(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import interrupt
        from puncture_agent.observability.tracing import (
            InMemoryTraceExporter,
            TraceRecorder,
        )

        scenarios = []

        def malformed_outcome(state, context):
            del state, context
            return {"not": "a NodeOutcome"}

        def raw_bytes(state, context):
            del context
            state.metadata["raw_payload"] = b"forbidden"
            return None

        def raw_interrupt(state, context):
            del state, context
            interrupt(b"forbidden")

        def cyclic_interrupt(state, context):
            del state, context
            value = []
            value.append(value)
            interrupt(value)

        scenarios.append(("malformed", malformed_outcome, "NODE_CONTRACT_ERROR"))
        scenarios.append(("raw-bytes", raw_bytes, "STATE_BOUNDARY_ERROR"))
        scenarios.append(("raw-interrupt", raw_interrupt, "STATE_BOUNDARY_ERROR"))
        scenarios.append(("cyclic-interrupt", cyclic_interrupt, "STATE_BOUNDARY_ERROR"))

        for suffix, handler, expected_code in scenarios:
            with self.subTest(suffix=suffix):
                calls = 0

                def counted_handler(state, context):
                    nonlocal calls
                    calls += 1
                    return handler(state, context)

                handlers = dict(build_mock_handlers())
                handlers["parse_request"] = counted_handler
                saver = InMemorySaver()
                exporter = InMemoryTraceExporter()
                session_id = f"real-langgraph-{suffix}-durable"
                state = AgentState(
                    user_query="对 Case-807 做路径规划",
                    session_id=session_id,
                )
                runtime = LangGraphRuntime(
                    PROJECT_ROOT / "graph" / "main_graph.json",
                    handlers,
                    checkpointer=saver,
                    tracer=TraceRecorder(exporter),
                )
                with self.assertRaises(GraphExecutionError):
                    runtime.run(state)
                self.assertEqual(AgentStatus.FAILED, state.status)
                self.assertEqual(expected_code, state.errors[-1]["code"])
                self.assertNotIn("raw_payload", state.metadata)
                node_spans = [
                    span for span in exporter.spans() if span.name == "agent.node"
                ]
                self.assertEqual("ERROR", node_spans[-1].status)
                restored = runtime.checkpoint_state(thread_id=session_id)
                self.assertEqual(state.to_dict(), restored.to_dict())
                resumed = runtime.resume(thread_id=session_id)
                self.assertEqual(AgentStatus.FAILED, resumed.status)
                self.assertEqual(1, calls)


if __name__ == "__main__":
    unittest.main()
