from __future__ import annotations

import unittest

from puncture_agent.runtime import (
    InMemoryRunService,
    IntegratedMockExecutor,
    RunRequest,
    RunStatus,
)


class IntegratedMockWorkflowTests(unittest.TestCase):
    def test_data_workflow_composes_model_rag_graph_tools_and_verifier(self) -> None:
        executor = IntegratedMockExecutor()
        service = InMemoryRunService(executor)
        snapshot = service.create_run(
            RunRequest(
                case_id="Case-203",
                user_query="检查 Case-203 的 MCS 标签和分割模型结果",
                task_type="DATA_MODEL_VALIDATION",
                idempotency_key="integration-data-1",
                tenant_id="tenant-a",
            )
        )

        self.assertEqual(RunStatus.SUCCEEDED, snapshot.status)
        self.assertIsNotNone(executor.last_model_response)
        self.assertIsNotNone(executor.last_rag_response)
        self.assertIsNotNone(executor.last_state)
        self.assertGreater(len(executor.last_rag_response.chunks), 0)
        self.assertGreater(len(executor.last_state.tool_calls), 0)
        self.assertEqual("PASS", executor.last_state.verification_status)
        self.assertGreater(snapshot.final_report["runtime_evidence"]["visited_node_count"], 0)

    def test_planning_workflow_keeps_deterministic_safety_verifier(self) -> None:
        executor = IntegratedMockExecutor()
        service = InMemoryRunService(executor)
        snapshot = service.create_run(
            RunRequest(
                case_id="Case-102",
                user_query="请对 Case-102 做路径规划和皮肤穿透安全评估",
                task_type="PLANNING_SAFETY",
                idempotency_key="integration-plan-1",
                tenant_id="tenant-a",
                metadata={"access_scopes": ["public", "algorithm_team"]},
            )
        )

        self.assertEqual(RunStatus.SUCCEEDED, snapshot.status)
        self.assertEqual("PASS", snapshot.final_report["verification_status"])
        self.assertEqual(["path-001", "path-002"], snapshot.final_report["safety_result"]["accepted_candidate_ids"])
        self.assertTrue(snapshot.final_report["skin_penetration_result"]["penetrated"])
        events = service.get_events(snapshot.run_id, tenant_id="tenant-a")
        node_names = [event.node_name for event in events]
        self.assertIn("model_gateway.plan", node_names)
        self.assertIn("rag.retrieve", node_names)
        self.assertIn("agent.graph", node_names)


if __name__ == "__main__":
    unittest.main()
