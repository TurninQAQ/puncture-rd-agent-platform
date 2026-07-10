from __future__ import annotations

import unittest
from dataclasses import fields

from puncture_agent.runtime.models import RunEvent, RunRequest, RunSnapshot


class ApiContractTests(unittest.TestCase):
    def test_run_request_field_names_are_frozen(self) -> None:
        self.assertEqual(
            [
                "case_id",
                "user_query",
                "task_type",
                "idempotency_key",
                "tenant_id",
                "principal_id",
                "artifact_ids",
                "metadata",
            ],
            [item.name for item in fields(RunRequest)],
        )

    def test_run_event_has_ordering_and_trace_fields(self) -> None:
        names = {item.name for item in fields(RunEvent)}
        self.assertTrue({"run_id", "sequence", "event_type", "node_name", "timestamp", "payload", "trace_id"} <= names)

    def test_snapshot_never_has_internal_uri_field(self) -> None:
        self.assertNotIn("internal_uri", {item.name for item in fields(RunSnapshot)})


if __name__ == "__main__":
    unittest.main()
