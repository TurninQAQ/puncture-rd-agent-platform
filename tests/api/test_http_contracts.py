from __future__ import annotations

import importlib.util
import unittest

from puncture_agent.api.privacy import (
    PublicValueValidationError,
    REDACTED,
    REDACTED_BINARY,
    redact_public_value,
    validate_public_json_input,
)
from puncture_agent.runtime import (
    EventType,
    RunEvent,
    RunRequest,
    RunServiceError,
    RunSnapshot,
    RunStatus,
)


PYDANTIC_AVAILABLE = importlib.util.find_spec("pydantic") is not None


class PublicApiPrivacyTests(unittest.TestCase):
    def test_redaction_is_recursive_json_safe_and_does_not_mutate_input(self) -> None:
        source = {
            "safe": {"count": 2},
            "authorization": "Bearer private-token",
            "nested": {
                "storage_uri": "s3://private-bucket/object",
                "binary": b"raw-image",
                "opaque": object(),
                "provider_value": "dependency failed at postgresql://user:secret@db/private",
                "provider_note": "upstream returned token abcdef123",
                "token_equals_note": "token=abcdef123",
                "token_colon_note": "token:abcdef123",
                "systemPrompt": "private instructions",
                "sourcePatientId": "patient-123",
                "sourcePatientID": "patient-456",
                "patientMRN": "mrn-789",
                "artifactInternalPath": "/private/object",
                "internalURI": "opaque-private-location",
                "network_note": "failed at https://service.internal/private",
                "pacs_note": "pacs://server/study",
                "single_scheme_note": "x://internal/path",
                "jwt_note": (
                    "prefix "
                    "abcdefghijkl.mnopqrstuvwx.yzABCDEFGHIJ"
                    " suffix"
                ),
            },
        }

        public = redact_public_value(source)

        self.assertEqual({"count": 2}, public["safe"])
        self.assertEqual(REDACTED, public["authorization"])
        self.assertEqual(REDACTED, public["nested"]["storage_uri"])
        self.assertEqual(REDACTED_BINARY, public["nested"]["binary"])
        self.assertEqual(REDACTED, public["nested"]["opaque"])
        self.assertEqual(REDACTED, public["nested"]["provider_value"])
        self.assertEqual(REDACTED, public["nested"]["provider_note"])
        self.assertEqual(REDACTED, public["nested"]["token_equals_note"])
        self.assertEqual(REDACTED, public["nested"]["token_colon_note"])
        self.assertEqual(REDACTED, public["nested"]["systemPrompt"])
        self.assertEqual(REDACTED, public["nested"]["sourcePatientId"])
        self.assertEqual(REDACTED, public["nested"]["sourcePatientID"])
        self.assertEqual(REDACTED, public["nested"]["patientMRN"])
        self.assertEqual(REDACTED, public["nested"]["artifactInternalPath"])
        self.assertEqual(REDACTED, public["nested"]["internalURI"])
        self.assertEqual(REDACTED, public["nested"]["network_note"])
        self.assertEqual(REDACTED, public["nested"]["pacs_note"])
        self.assertEqual(REDACTED, public["nested"]["single_scheme_note"])
        self.assertEqual(REDACTED, public["nested"]["jwt_note"])
        self.assertEqual("Bearer private-token", source["authorization"])
        self.assertEqual(b"raw-image", source["nested"]["binary"])

    def test_public_json_validation_rejects_binary_raw_images_and_nonfinite(self) -> None:
        validate_public_json_input(
            {"constraints": {"entry_point": [1.0, 2.0, 3.0]}}
        )
        for value in (
            {"payload": b"forbidden"},
            {"nested": {"voxels": [0, 1]}},
            {"nested": {"ct_pixels": [0, 1]}},
            {"nested": {"pixelData": [0, 1]}},
            {"nested": {"dicomVoxels": [0, 1]}},
            {"metadata": {"authorization_header": "private"}},
            {"endpoint": "failed at postgresql://user:secret@db/private"},
            {"endpoint": "http://10.0.0.1/private"},
            {"endpoint": "pacs://server/study"},
            {"endpoint": "x://internal/path"},
            {"note": "token=abcdef123"},
            {"note": "token:abcdef123"},
            {
                "note": (
                    "prefix "
                    "abcdefghijkl.mnopqrstuvwx.yzABCDEFGHIJ"
                    " suffix"
                )
            },
            {"score": float("nan")},
            {"keys": {1: "not-json"}},
        ):
            with self.subTest(value=value):
                with self.assertRaises(PublicValueValidationError):
                    validate_public_json_input(value)
        with self.assertRaisesRegex(PublicValueValidationError, "depth limit"):
            validate_public_json_input({"one": {"two": True}}, max_depth=1)
        with self.assertRaisesRegex(PublicValueValidationError, "node limit"):
            validate_public_json_input([1, 2], max_nodes=2)


@unittest.skipUnless(PYDANTIC_AVAILABLE, "Pydantic implementation dependency is not installed")
class PydanticHttpContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from puncture_agent.api.http_contracts import (
            ApiRequestValidationError,
            ApprovalBody,
            AuthenticatedPrincipal,
            RunCreateBody,
            RunEventResponse,
            RunSnapshotResponse,
            map_exception_to_http_error,
        )

        cls.ApprovalBody = ApprovalBody
        cls.ApiRequestValidationError = ApiRequestValidationError
        cls.AuthenticatedPrincipal = AuthenticatedPrincipal
        cls.RunCreateBody = RunCreateBody
        cls.RunEventResponse = RunEventResponse
        cls.RunSnapshotResponse = RunSnapshotResponse
        cls.map_exception_to_http_error = staticmethod(map_exception_to_http_error)

    def test_authenticated_identity_is_injected_not_accepted_from_body(self) -> None:
        from pydantic import ValidationError

        body = self.RunCreateBody(
            case_id="Case-001",
            user_query="validate this case",
            task_type="DATA_MODEL_VALIDATION",
            idempotency_key="create-001",
            artifact_ids=["artifact-001", "artifact-002"],
        )
        request = body.to_runtime_request(
            self.AuthenticatedPrincipal(
                " tenant-a ",
                " principal-a ",
                ("public", "algorithm_team"),
            )
        )

        self.assertEqual("tenant-a", request.tenant_id)
        self.assertEqual("principal-a", request.principal_id)
        self.assertEqual(("artifact-001", "artifact-002"), request.artifact_ids)
        self.assertEqual(
            ["public", "algorithm_team"],
            request.metadata["access_scopes"],
        )
        with self.assertRaises(ValidationError):
            self.RunCreateBody.model_validate(
                {
                    **body.model_dump(mode="json"),
                    "tenant_id": "spoofed-tenant",
                }
            )

    def test_request_validation_rejects_bad_contract_and_raw_payloads(self) -> None:
        from pydantic import ValidationError

        valid = {
            "case_id": "Case-002",
            "user_query": "plan a path",
            "task_type": "PLANNING_SAFETY",
            "idempotency_key": "create-002",
        }
        invalid_values = (
            {**valid, "case_id": ""},
            {**valid, "case_id": "../escape"},
            {**valid, "task_type": "UNKNOWN"},
            {**valid, "user_query": "use token abcdef123"},
            {**valid, "idempotency_key": "bad key"},
            {**valid, "artifact_ids": ["bad artifact id"]},
            {**valid, "artifact_ids": ["artifact-1", "artifact-1"]},
            {**valid, "metadata": {"pixels": [0, 1, 2]}},
            {**valid, "metadata": {"access_token": "private"}},
            {**valid, "metadata": {"value": float("inf")}},
            {**valid, "unknown": True},
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises((ValidationError, PublicValueValidationError)):
                    self.RunCreateBody.model_validate(value)

    def test_request_schema_and_json_round_trip_are_locked(self) -> None:
        import pydantic

        self.assertEqual(2, int(pydantic.__version__.split(".", 1)[0]))
        schema = self.RunCreateBody.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            ["case_id", "user_query", "task_type", "idempotency_key"],
            schema["required"],
        )
        self.assertEqual(
            ["DATA_MODEL_VALIDATION", "PLANNING_SAFETY"],
            schema["properties"]["task_type"]["enum"],
        )
        self.assertEqual(128, schema["properties"]["artifact_ids"]["maxItems"])
        body = self.RunCreateBody(
            case_id="Case-002",
            user_query="plan a path",
            task_type="PLANNING_SAFETY",
            idempotency_key="create-json-round-trip",
            metadata={"constraints": {"top_k": 3}},
        )
        restored = self.RunCreateBody.model_validate_json(body.model_dump_json())
        self.assertEqual(body, restored)

    def test_test_only_controls_are_disabled_unless_explicitly_allowed(self) -> None:
        body = self.RunCreateBody(
            case_id="Case-003",
            user_query="validate",
            task_type="DATA_MODEL_VALIDATION",
            idempotency_key="create-003",
            metadata={"dependency_timeout": True},
        )
        principal = self.AuthenticatedPrincipal("tenant-a", "principal-a")

        with self.assertRaisesRegex(ValueError, "test-only"):
            body.to_runtime_request(principal)
        request = body.to_runtime_request(principal, allow_test_controls=True)
        self.assertTrue(request.metadata["dependency_timeout"])

        for key, value in (
            ("access_scopes", ["admin"]),
            ("accessScopes", ["admin"]),
            ("accessSCOPES", ["admin"]),
            ("roles", ["admin"]),
            ("tenant_id", "other"),
            ("tenantId", "other"),
            ("principal_id", "other"),
            ("nested", {"projectIds": ["private-project"]}),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    (ValueError, PublicValueValidationError),
                    "server-owned|authenticated authority",
                ):
                    self.RunCreateBody(
                        case_id="Case-003",
                        user_query="validate",
                        task_type="DATA_MODEL_VALIDATION",
                        idempotency_key=f"create-003-{key}",
                        metadata={key: value},
                    ).to_runtime_request(
                        principal,
                        allow_test_controls=True,
                    )

    def test_approval_body_uses_authenticated_principal_and_strict_boolean(self) -> None:
        from pydantic import ValidationError

        body = self.ApprovalBody(approved=True, comment="reviewed")
        decision = body.to_runtime_decision(
            approval_id="approval-1",
            principal=self.AuthenticatedPrincipal("tenant-a", "reviewer-a"),
        )

        self.assertEqual("reviewer-a", decision.principal_id)
        self.assertTrue(decision.approved)
        with self.assertRaises(ValidationError):
            self.ApprovalBody(approved="true")
        with self.assertRaisesRegex(ValueError, "approval_id"):
            body.to_runtime_decision(
                approval_id="../forbidden",
                principal=self.AuthenticatedPrincipal("tenant-a", "reviewer-a"),
            )

    def test_snapshot_and_event_views_redact_every_public_mapping(self) -> None:
        request = RunRequest(
            case_id="Case-004",
            user_query="validate",
            task_type="DATA_MODEL_VALIDATION",
            idempotency_key="create-004",
            tenant_id="tenant-a",
            principal_id="principal-a",
            metadata={"token": "secret", "safe": {"value": 1}},
        )
        snapshot = RunSnapshot(
            run_id="run-004",
            request=request,
            status=RunStatus.FAILED,
            trace_id="trace-004",
            created_at="2026-07-11T00:00:00.000Z",
            updated_at="2026-07-11T00:00:01.000Z",
            final_report={"internal_uri": "file:///private/report", "ok": False},
            checkpoint={"raw": b"binary", "node": "parse_request"},
            approval_id=None,
            error={"code": "FAILED", "message": "safe", "api_key": "private"},
        )
        event_payload = {"patient_name": "Private", "safe": [1, 2]}
        event = RunEvent(
            run_id="run-004",
            sequence=1,
            event_type=EventType.RUN_FAILED,
            node_name=None,
            timestamp="2026-07-11T00:00:01.000Z",
            payload=event_payload,
            trace_id="trace-004",
        )

        public_snapshot = self.RunSnapshotResponse.from_runtime(snapshot)
        public_event = self.RunEventResponse.from_runtime(event)

        self.assertEqual(REDACTED, public_snapshot.request.metadata["token"])
        self.assertEqual(REDACTED, public_snapshot.request.user_query)
        self.assertEqual(REDACTED, public_snapshot.request.idempotency_key)
        self.assertEqual(REDACTED, public_snapshot.final_report["internal_uri"])
        self.assertEqual(REDACTED_BINARY, public_snapshot.checkpoint["raw"])
        self.assertEqual(REDACTED, public_snapshot.error["api_key"])
        self.assertEqual(REDACTED, public_snapshot.error["message"])
        self.assertEqual(REDACTED, public_event.payload["patient_name"])
        self.assertEqual([1, 2], public_event.payload["safe"])
        snapshot.final_report["ok"] = True
        event_payload["safe"].append(3)
        self.assertFalse(public_snapshot.final_report["ok"])
        self.assertEqual([1, 2], public_event.payload["safe"])

    def test_exception_mapping_is_stable_and_hides_unknown_details(self) -> None:
        not_found = self.map_exception_to_http_error(
            RunServiceError("NOT_FOUND", "run was not found")
        )
        retryable = self.map_exception_to_http_error(
            RunServiceError(
                "CHECKPOINT_UNAVAILABLE",
                "postgresql://user:secret@db/private",
                retryable=True,
            )
        )
        invalid = self.map_exception_to_http_error(
            self.ApiRequestValidationError("private input")
        )
        unknown = self.map_exception_to_http_error(RuntimeError("private stack"))
        internal_value_error = self.map_exception_to_http_error(
            ValueError("private code defect")
        )

        self.assertEqual(404, not_found.status_code)
        self.assertEqual("NOT_FOUND", not_found.response.error.code)
        self.assertEqual("resource was not found", not_found.response.error.message)
        self.assertEqual(503, retryable.status_code)
        self.assertEqual("SERVICE_UNAVAILABLE", retryable.response.error.code)
        self.assertEqual(
            "service dependency is unavailable",
            retryable.response.error.message,
        )
        self.assertEqual("INVALID_REQUEST", invalid.response.error.code)
        self.assertEqual("request validation failed", invalid.response.error.message)
        self.assertEqual(500, unknown.status_code)
        self.assertEqual("internal service error", unknown.response.error.message)
        self.assertEqual(500, internal_value_error.status_code)
        self.assertEqual("INTERNAL_ERROR", internal_value_error.response.error.code)

        private_service_error = self.map_exception_to_http_error(
            RunServiceError("PRIVATE_BACKEND_ERROR", "private stack")
        )
        self.assertEqual(500, private_service_error.status_code)
        self.assertEqual("INTERNAL_ERROR", private_service_error.response.error.code)


if __name__ == "__main__":
    unittest.main()
