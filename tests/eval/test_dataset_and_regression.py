from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent import (  # noqa: E402
    AgentStatus,
    GraphRuntime,
    TaskType,
    build_mock_handlers,
)
from puncture_agent.agent.state import AgentState  # noqa: E402
from puncture_agent.observability.dataset import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    DatasetValidationError,
    dump_eval_dataset,
    load_eval_dataset,
)
from puncture_agent.observability.eval_harness import (  # noqa: E402
    AgentEvalHarness,
    EvalCase,
    EvalCaseResult,
    EvalReport,
    compare_reports,
    evaluate_case,
)


FIXTURES = PROJECT_ROOT / "tests" / "eval" / "fixtures"


def runtime_factory() -> GraphRuntime:
    return GraphRuntime(
        PROJECT_ROOT / "graph" / "main_graph.json",
        build_mock_handlers(),
    )


class DatasetLoaderTests(unittest.TestCase):
    def test_load_json_dataset(self) -> None:
        schema, version, cases = load_eval_dataset(FIXTURES / "mock_reference_v1.json")
        self.assertEqual(DATASET_SCHEMA_VERSION, schema)
        self.assertEqual("mock-reference-v1", version)
        self.assertEqual(3, len(cases))
        self.assertEqual("planning-timeout-recovery", cases[2].case_id)
        self.assertTrue(cases[2].metadata.get("is_recovery_case"))
        self.assertEqual(
            ["generate_candidate_paths"],
            cases[2].metadata.get("fail_tool_once"),
        )

    def test_load_jsonl_dataset(self) -> None:
        schema, version, cases = load_eval_dataset(FIXTURES / "mock_reference_v1.jsonl")
        self.assertEqual(DATASET_SCHEMA_VERSION, schema)
        self.assertEqual("mock-reference-v1-jsonl", version)
        self.assertEqual(2, len(cases))

    def test_unsupported_schema_and_duplicate_ids_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad_schema = Path(directory) / "bad.json"
            bad_schema.write_text(
                json.dumps(
                    {
                        "schema_version": "eval-case-v0",
                        "dataset_version": "x",
                        "cases": [
                            {
                                "case_id": "a",
                                "query": "q",
                                "expected_task_type": "PLANNING_SAFETY",
                                "expected_terminal_status": "SUCCEEDED",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(DatasetValidationError):
                load_eval_dataset(bad_schema)

            dup = Path(directory) / "dup.json"
            case = {
                "case_id": "same",
                "query": "q",
                "expected_task_type": "PLANNING_SAFETY",
                "expected_terminal_status": "SUCCEEDED",
            }
            dump_eval_dataset(
                [
                    EvalCase(**{**case, "case_id": "same"}),
                ],
                dup,
                dataset_version="d",
            )
            # Hand-write duplicates because dump dedupes by construction.
            payload = json.loads(dup.read_text(encoding="utf-8"))
            payload["cases"].append(dict(payload["cases"][0]))
            dup.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                load_eval_dataset(dup)

            unknown = Path(directory) / "unknown.json"
            unknown.write_text(
                json.dumps(
                    {
                        "schema_version": "eval-case-v1",
                        "dataset_version": "x",
                        "cases": [
                            {
                                "case_id": "a",
                                "query": "q",
                                "expected_task_type": "PLANNING_SAFETY",
                                "expected_terminal_status": "SUCCEEDED",
                                "not_a_real_field": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(DatasetValidationError):
                load_eval_dataset(unknown)

    def test_empty_dataset_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty.json"
            empty.write_text(
                json.dumps(
                    {
                        "schema_version": "eval-case-v1",
                        "dataset_version": "empty",
                        "cases": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(DatasetValidationError):
                load_eval_dataset(empty)

    def test_round_trip_dump_load(self) -> None:
        _, _, cases = load_eval_dataset(FIXTURES / "mock_reference_v1.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round.json"
            dump_eval_dataset(cases[:1], path, dataset_version="round-v1")
            schema, version, loaded = load_eval_dataset(path)
            self.assertEqual(DATASET_SCHEMA_VERSION, schema)
            self.assertEqual("round-v1", version)
            self.assertEqual(cases[0].case_id, loaded[0].case_id)


class ExtendedHarnessTests(unittest.TestCase):
    def test_fixture_dataset_passes_contract_metrics(self) -> None:
        schema, version, cases = load_eval_dataset(FIXTURES / "mock_reference_v1.json")
        self.assertEqual(DATASET_SCHEMA_VERSION, schema)
        report = AgentEvalHarness(runtime_factory).run(cases, dataset_version=version)
        self.assertEqual(3, report.case_count)
        self.assertEqual(3, report.passed_case_count)
        self.assertEqual(1.0, report.metrics["task_success_rate"])
        self.assertEqual(0.0, report.metrics["forbidden_node_violation_rate"])
        self.assertEqual(1.0, report.metrics["tool_parameter_validity_rate"])
        self.assertEqual(1.0, report.metrics["retry_recovery_rate"])
        self.assertFalse(report.release_blocked)
        # Serialization is stable for non-timing fields.
        payload = report.to_dict()
        self.assertEqual("eval-report-v1", payload["schema_version"])
        self.assertEqual(version, payload["dataset_version"])
        encoded = report.to_json()
        self.assertIn("task_success_rate", encoded)

    def test_failed_case_diagnostics_include_observed_path(self) -> None:
        case = EvalCase(
            case_id="intentional-failure",
            query="对 Case-703 做路径规划",
            expected_task_type=TaskType.DATA_MODEL_VALIDATION,
            expected_terminal_status=AgentStatus.FAILED,
            required_nodes=("convert_mcs_to_nifti",),
            expected_tools=("run_segmentation",),
            minimum_citations=10,
        )
        report = AgentEvalHarness(runtime_factory).run([case])
        self.assertEqual(0, report.passed_case_count)
        result = report.cases[0]
        self.assertGreaterEqual(len(result.failures), 4)
        self.assertTrue(result.diagnostics)
        self.assertIn("visited_nodes", result.observed)
        self.assertIn("called_tools", result.observed)
        self.assertIn("status", result.observed)

    def test_tool_parameter_predicate_failure_is_diagnosed(self) -> None:
        state = AgentState(user_query="x", task_type=TaskType.PLANNING_SAFETY, status=AgentStatus.SUCCEEDED)
        state.visited_nodes = ["generate_candidate_paths"]
        state.tool_calls = [
            {
                "tool_name": "generate_candidate_paths",
                "request": {"max_needle_length_mm": 50.0},
            }
        ]
        state.final_report = {
            "report_version": "1",
            "status": "SUCCEEDED",
            "case_id": "c",
        }
        state.citations = [{"id": "1"}, {"id": "2"}]
        case = EvalCase(
            case_id="param-fail",
            query="x",
            expected_task_type=TaskType.PLANNING_SAFETY,
            expected_terminal_status=AgentStatus.SUCCEEDED,
            expected_tools=("generate_candidate_paths",),
            minimum_citations=2,
            metadata={
                "expected_tool_argument_predicates": [
                    {
                        "tool_name": "generate_candidate_paths",
                        "field": "max_needle_length_mm",
                        "operator": "eq",
                        "value": 120.0,
                    }
                ]
            },
        )
        result = evaluate_case(case, state)
        self.assertFalse(result.passed)
        self.assertTrue(any("tool argument predicates" in item for item in result.failures))

    def test_current_version_and_acl_failures(self) -> None:
        state = AgentState(
            user_query="x",
            task_type=TaskType.PLANNING_SAFETY,
            status=AgentStatus.SUCCEEDED,
            final_report={
                "report_version": "1",
                "status": "SUCCEEDED",
                "case_id": "c",
            },
            citations=[{"id": "1"}, {"id": "2"}],
        )
        state.retrieved_documents = [
            {"document_id": "doc-old", "version": "v1", "chunk_id": "c1"}
        ]
        state.metadata["acl_violation_count"] = 1
        case = EvalCase(
            case_id="version-acl",
            query="x",
            expected_task_type=TaskType.PLANNING_SAFETY,
            expected_terminal_status=AgentStatus.SUCCEEDED,
            minimum_citations=2,
            metadata={"expected_document_version": "v9"},
        )
        result = evaluate_case(case, state)
        self.assertFalse(result.passed)
        self.assertTrue(any("document version" in item for item in result.failures))
        self.assertTrue(any("ACL" in item for item in result.failures))
        self.assertEqual(1, result.acl_violation_count)
        self.assertFalse(result.current_version_hit)


class RegressionReportTests(unittest.TestCase):
    def _result(self, case_id: str, passed: bool, **kwargs: object) -> EvalCaseResult:
        return EvalCaseResult(
            case_id=case_id,
            passed=passed,
            routing_correct=passed,
            terminal_status_correct=passed,
            required_node_coverage=1.0 if passed else 0.0,
            forbidden_node_violation_count=int(kwargs.get("forbidden", 0)),
            tool_recall=1.0 if passed else 0.0,
            citation_requirement_met=True,
            schema_valid=True,
            failures=() if passed else ("failed",),
            observed={"status": "SUCCEEDED" if passed else "FAILED"},
            acl_violation_count=int(kwargs.get("acl", 0)),
            diagnostics=() if passed else ("failed",),
        )

    def test_classifies_improved_regressed_and_blocks_safety(self) -> None:
        baseline = EvalReport(
            case_count=3,
            passed_case_count=2,
            metrics={
                "task_success_rate": 2.0 / 3.0,
                "forbidden_node_violation_rate": 0.0,
                "acl_violation_rate": 0.0,
            },
            cases=(
                self._result("a", True),
                self._result("b", False),
                self._result("c", True),
            ),
            dataset_version="baseline",
        )
        candidate = EvalReport(
            case_count=3,
            passed_case_count=2,
            metrics={
                "task_success_rate": 2.0 / 3.0,
                "forbidden_node_violation_rate": 0.5,
                "acl_violation_rate": 0.0,
            },
            cases=(
                self._result("a", False),  # regressed
                self._result("b", True),  # improved
                self._result("d", True),  # new; c removed
            ),
            dataset_version="candidate",
            release_blocked=True,
            release_block_reasons=("forbidden_node_violation_rate > 0",),
        )
        report = compare_reports(baseline, candidate)
        self.assertEqual(("b",), report.improved_case_ids)
        self.assertEqual(("a",), report.regressed_case_ids)
        self.assertEqual(("d",), report.new_case_ids)
        self.assertEqual(("c",), report.removed_case_ids)
        self.assertTrue(report.safety_regression)
        self.assertTrue(report.release_blocked)
        self.assertIn("task_success_rate", report.metric_deltas)
        encoded = report.to_json()
        self.assertIn("regressed_case_ids", encoded)
        # Deterministic key order via sort_keys.
        self.assertEqual(encoded, report.to_json())


if __name__ == "__main__":
    unittest.main()
