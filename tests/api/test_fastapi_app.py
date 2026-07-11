from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import importlib.util
import json
import os
import unittest
from uuid import uuid4

from contracts.artifacts import ArtifactPublicView
from contracts.enums import ArtifactStatus, ArtifactType
from puncture_agent.api.body_admission import RawBodyAdmissionMiddleware
from puncture_agent.runtime import InMemoryRunService, RunServiceError, ScenarioExecutor


FASTAPI_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("fastapi", "httpx", "pydantic", "starlette")
)
POSTGRES_DSN = os.environ.get("PUNCTURE_TEST_POSTGRES_DSN", "")


async def _exercise_admission(
    *,
    headers: list[tuple[bytes, bytes]],
    messages: list[dict],
    max_body_bytes: int = 8,
    path: str = "/api/v1/runs",
) -> tuple[list[dict], list[bytes]]:
    sent: list[dict] = []
    downstream_bodies: list[bytes] = []
    queue = list(messages)

    async def receive() -> dict:
        if queue:
            return queue.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    async def downstream(scope, replay_receive, replay_send) -> None:
        del scope
        message = await replay_receive()
        downstream_bodies.append(message.get("body", b""))
        await replay_send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [],
            }
        )
        await replay_send({"type": "http.response.body", "body": b""})

    middleware = RawBodyAdmissionMiddleware(
        downstream,
        max_body_bytes=max_body_bytes,
    )
    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": headers,
        },
        receive,
        send,
    )
    return sent, downstream_bodies


class RawBodyAdmissionTests(unittest.TestCase):
    def test_rejects_duplicate_length_before_downstream(self) -> None:
        sent, downstream = asyncio.run(
            _exercise_admission(
                headers=[
                    (b"content-type", b"application/json"),
                    (b"content-length", b"2"),
                    (b"content-length", b"2"),
                ],
                messages=[
                    {"type": "http.request", "body": b"{}", "more_body": False}
                ],
            )
        )

        self.assertEqual(400, sent[0]["status"])
        self.assertEqual([], downstream)

    def test_rejects_chunked_overflow_and_forged_length(self) -> None:
        overflow, overflow_downstream = asyncio.run(
            _exercise_admission(
                headers=[(b"content-type", b"application/json")],
                messages=[
                    {"type": "http.request", "body": b"12345", "more_body": True},
                    {"type": "http.request", "body": b"6789", "more_body": False},
                ],
            )
        )
        forged, forged_downstream = asyncio.run(
            _exercise_admission(
                headers=[
                    (b"content-type", b"application/json"),
                    (b"content-length", b"1"),
                ],
                messages=[
                    {"type": "http.request", "body": b"{}", "more_body": False}
                ],
            )
        )
        giant, giant_downstream = asyncio.run(
            _exercise_admission(
                headers=[(b"content-type", b"application/json")],
                messages=[
                    {
                        "type": "http.request",
                        "body": b"x" * 100_000,
                        "more_body": False,
                    }
                ],
            )
        )
        huge_length, huge_length_downstream = asyncio.run(
            _exercise_admission(
                headers=[
                    (b"content-type", b"application/json"),
                    (b"content-length", b"9" * 100_000),
                ],
                messages=[
                    {"type": "http.request", "body": b"{}", "more_body": False}
                ],
            )
        )

        self.assertEqual(413, overflow[0]["status"])
        self.assertEqual([], overflow_downstream)
        self.assertEqual(400, forged[0]["status"])
        self.assertEqual([], forged_downstream)
        self.assertEqual(413, giant[0]["status"])
        self.assertEqual([], giant_downstream)
        self.assertEqual(400, huge_length[0]["status"])
        self.assertEqual([], huge_length_downstream)

    def test_rejects_compression_and_replays_exact_limit(self) -> None:
        compressed, compressed_downstream = asyncio.run(
            _exercise_admission(
                headers=[
                    (b"content-type", b"application/json"),
                    (b"content-encoding", b"gzip"),
                ],
                messages=[
                    {"type": "http.request", "body": b"{}", "more_body": False}
                ],
            )
        )
        accepted, accepted_downstream = asyncio.run(
            _exercise_admission(
                headers=[
                    (b"content-type", b"application/json"),
                    (b"content-length", b"8"),
                ],
                messages=[
                    {"type": "http.request", "body": b"12345678", "more_body": False}
                ],
            )
        )

        self.assertEqual(415, compressed[0]["status"])
        self.assertEqual([], compressed_downstream)
        self.assertEqual(204, accepted[0]["status"])
        self.assertEqual([b"12345678"], accepted_downstream)


if FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient

    from puncture_agent.api.fastapi_app import (
        ApiPermission,
        AuthorizedCase,
        create_app,
    )
    from puncture_agent.api.http_contracts import AuthenticatedPrincipal


    class _Authenticator:
        def __init__(self) -> None:
            self.principals = {
                "token-a": AuthenticatedPrincipal(
                    "tenant-a",
                    "principal-a",
                    ("rag-public", "case-reader"),
                ),
                "token-b": AuthenticatedPrincipal(
                    "tenant-b",
                    "principal-b",
                    ("rag-public",),
                ),
            }
            self.tokens: list[str] = []

        def authenticate(self, bearer_token: str) -> AuthenticatedPrincipal:
            self.tokens.append(bearer_token)
            try:
                return self.principals[bearer_token]
            except KeyError as exc:
                raise RunServiceError("FORBIDDEN", "invalid token") from exc


    class _Authorizer:
        def __init__(self) -> None:
            self.allowed_cases = {
                "tenant-a": {"Case-001", "Case-Approval", "Case-Failure"},
                "tenant-b": {"Case-001"},
            }
            self.permissions = set(ApiPermission)
            self.projects = {
                (tenant_id, case_id): f"project-{tenant_id}"
                for tenant_id, case_ids in self.allowed_cases.items()
                for case_id in case_ids
            }
            self.calls: list[tuple[str, str, ApiPermission]] = []

        def _require(
            self,
            principal: AuthenticatedPrincipal,
            case_id: str,
            permission: ApiPermission,
        ) -> AuthorizedCase:
            self.calls.append((principal.tenant_id, case_id, permission))
            if (
                permission not in self.permissions
                or case_id not in self.allowed_cases.get(principal.tenant_id, set())
            ):
                raise RunServiceError("FORBIDDEN", "access denied")
            return AuthorizedCase(
                tenant_id=principal.tenant_id,
                project_id=self.projects[(principal.tenant_id, case_id)],
                case_id=case_id,
            )

        def require_case(
            self,
            principal: AuthenticatedPrincipal,
            *,
            case_id: str,
            permission: ApiPermission,
        ) -> AuthorizedCase:
            return self._require(principal, case_id, permission)

        def require_run(
            self,
            principal: AuthenticatedPrincipal,
            *,
            snapshot,
            permission: ApiPermission,
        ) -> AuthorizedCase:
            return self._require(principal, snapshot.request.case_id, permission)


    class _ArtifactGateway:
        def __init__(self) -> None:
            self.records = {
                "artifact-001": ArtifactPublicView(
                    artifact_id="artifact-001",
                    case_id="Case-001",
                    artifact_type=ArtifactType.CT_VOLUME,
                    status=ArtifactStatus.AVAILABLE,
                    producer_name="case-loader",
                    producer_version="1.0",
                    geometry_fingerprint="geometry-public",
                )
            }
            self.calls: list[tuple[str, tuple[str, ...], ApiPermission]] = []
            self.denied_tenants: set[str] = set()

        def require_artifacts(
            self,
            principal: AuthenticatedPrincipal,
            *,
            case: AuthorizedCase,
            artifact_ids: tuple[str, ...],
            permission: ApiPermission,
        ) -> tuple[ArtifactPublicView, ...]:
            self.calls.append((principal.tenant_id, artifact_ids, permission))
            if principal.tenant_id in self.denied_tenants:
                raise RunServiceError("FORBIDDEN", "artifact access denied")
            try:
                return tuple(self.records[item] for item in artifact_ids)
            except KeyError as exc:
                raise RunServiceError("NOT_FOUND", "artifact not found") from exc

        def get_metadata(
            self,
            principal: AuthenticatedPrincipal,
            *,
            artifact_id: str,
            permission: ApiPermission,
        ) -> ArtifactPublicView:
            self.calls.append((principal.tenant_id, (artifact_id,), permission))
            if principal.tenant_id in self.denied_tenants:
                raise RunServiceError("FORBIDDEN", "artifact access denied")
            try:
                return self.records[artifact_id]
            except KeyError as exc:
                raise RunServiceError("NOT_FOUND", "artifact not found") from exc


    class _FailingHealthProbe:
        def status(self) -> str:
            raise OSError("postgresql://user:secret@db/private")


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI implementation dependencies are not installed")
class FastApiTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = ScenarioExecutor()
        self.service = InMemoryRunService(self.executor)
        self.authenticator = _Authenticator()
        self.authorizer = _Authorizer()
        self.artifacts = _ArtifactGateway()
        self.stack = ExitStack()
        self.client = self._client()

    def tearDown(self) -> None:
        self.stack.close()

    def _client(self, **kwargs) -> "TestClient":
        app = create_app(
            kwargs.pop("run_service", self.service),
            authenticator=kwargs.pop("authenticator", self.authenticator),
            authorizer=kwargs.pop("authorizer", self.authorizer),
            artifact_gateway=kwargs.pop("artifact_gateway", self.artifacts),
            allow_test_controls=kwargs.pop("allow_test_controls", True),
            **kwargs,
        )
        return self.stack.enter_context(
            TestClient(app, raise_server_exceptions=False)
        )

    @staticmethod
    def _headers(token: str = "token-a") -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _body(
        *,
        case_id: str = "Case-001",
        key: str = "create-001",
        metadata: dict | None = None,
        artifacts: bool = True,
    ) -> dict:
        return {
            "case_id": case_id,
            "user_query": "validate the case",
            "task_type": "DATA_MODEL_VALIDATION",
            "idempotency_key": key,
            "artifact_ids": ["artifact-001"] if artifacts else [],
            "metadata": metadata or {},
        }

    def test_openapi_has_exact_endpoints_security_and_public_artifact_schema(self) -> None:
        document = self.client.get("/openapi.json").json()
        expected = {
            "/api/v1/runs": {"post"},
            "/api/v1/runs/{run_id}": {"get"},
            "/api/v1/runs/{run_id}/events": {"get"},
            "/api/v1/runs/{run_id}/approvals/{approval_id}": {"post"},
            "/api/v1/runs/{run_id}/cancel": {"post"},
            "/api/v1/runs/{run_id}/resume": {"post"},
            "/api/v1/artifacts/{artifact_id}/metadata": {"get"},
            "/health": {"get"},
            "/metrics": {"get"},
        }

        self.assertEqual(expected, {path: set(item) for path, item in document["paths"].items()})
        self.assertIn("BearerAuth", document["components"]["securitySchemes"])
        for path in expected:
            operation = next(iter(expected[path]))
            security = document["paths"][path][operation].get("security", [])
            if path.startswith("/api/"):
                self.assertEqual([{"BearerAuth": []}], security)
            else:
                self.assertEqual([], security)
        artifact_schema = document["components"]["schemas"]["ArtifactMetadataResponse"]
        self.assertEqual(
            {
                "artifact_id",
                "case_id",
                "artifact_type",
                "status",
                "producer_name",
                "producer_version",
                "geometry_fingerprint",
            },
            set(artifact_schema["properties"]),
        )
        self.assertFalse(
            {"uri", "checksum_sha256", "metadata", "parent_artifact_ids"}
            & set(artifact_schema["properties"])
        )

    def test_postgres_settings_hide_dsn_and_validate_environment(self) -> None:
        from puncture_agent.api.postgres_app import PostgresApiSettings

        settings = PostgresApiSettings(
            connection_string="postgresql://user:private@db/agent"
        )

        self.assertNotIn("private", repr(settings))
        self.assertNotIn("private", str(settings))
        with self.assertRaisesRegex(ValueError, "PUNCTURE_API_POSTGRES_DSN"):
            PostgresApiSettings.from_env({})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PostgresApiSettings.from_env(
                {
                    "PUNCTURE_API_POSTGRES_DSN": "postgresql://db/agent",
                    "PUNCTURE_API_POSTGRES_LOCK_TIMEOUT_MS": "0",
                }
            )
        with self.assertRaises(ValueError) as invalid_float:
            PostgresApiSettings.from_env(
                {
                    "PUNCTURE_API_POSTGRES_DSN": "postgresql://db/agent",
                    "PUNCTURE_API_POSTGRES_CONNECT_TIMEOUT_SECONDS": "private-dsn",
                }
            )
        self.assertNotIn("private-dsn", str(invalid_float.exception))

    def test_create_get_event_replay_artifact_metadata_and_metrics(self) -> None:
        created = self.client.post(
            "/api/v1/runs",
            headers=self._headers(),
            json=self._body(),
        )

        self.assertEqual(200, created.status_code, created.text)
        snapshot = created.json()
        run_id = snapshot["run_id"]
        self.assertEqual("tenant-a", snapshot["request"]["tenant_id"])
        self.assertEqual("principal-a", snapshot["request"]["principal_id"])
        self.assertEqual("[REDACTED]", snapshot["request"]["user_query"])
        self.assertEqual("[REDACTED]", snapshot["request"]["idempotency_key"])
        self.assertEqual("[REDACTED]", snapshot["request"]["metadata"]["access_scopes"])
        self.assertEqual("[REDACTED]", snapshot["request"]["metadata"]["project_id"])
        self.assertEqual(1, self.executor.execution_count)
        self.assertEqual(ApiPermission.RUN_CREATE, self.authorizer.calls[0][2])
        self.assertEqual(ApiPermission.ARTIFACT_USE, self.artifacts.calls[0][2])

        fetched = self.client.get(
            f"/api/v1/runs/{run_id}",
            headers=self._headers(),
        )
        events = self.client.get(
            f"/api/v1/runs/{run_id}/events?after_sequence=2",
            headers=self._headers(),
        )
        artifact = self.client.get(
            "/api/v1/artifacts/artifact-001/metadata",
            headers=self._headers(),
        )

        self.assertEqual(200, fetched.status_code)
        self.assertEqual(200, events.status_code)
        sequences = [item["sequence"] for item in events.json()]
        self.assertEqual(sorted(sequences), sequences)
        self.assertTrue(all(sequence > 2 for sequence in sequences))
        self.assertEqual(200, artifact.status_code)
        self.assertEqual(
            {
                "artifact_id",
                "case_id",
                "artifact_type",
                "status",
                "producer_name",
                "producer_version",
                "geometry_fingerprint",
            },
            set(artifact.json()),
        )

        metrics = self.client.get("/metrics")
        self.assertEqual(200, metrics.status_code)
        self.assertIn('/api/v1/runs/{run_id}', metrics.text)
        for private_value in (run_id, "Case-001", "artifact-001", "token-a", "principal-a"):
            self.assertNotIn(private_value, metrics.text)

    def test_authentication_authorization_and_tenant_isolation_fail_closed(self) -> None:
        missing = self.client.post("/api/v1/runs", json=self._body())
        malformed_anonymous = self.client.post(
            "/api/v1/runs",
            headers={"Content-Type": "application/json"},
            content=b"{",
        )
        duplicate_auth = self.client.post(
            "/api/v1/runs",
            headers=[
                ("Authorization", "Bearer token-a"),
                ("Authorization", "Bearer token-a"),
            ],
            json=self._body(key="duplicate-auth"),
        )
        spoofed = self.client.post(
            "/api/v1/runs",
            headers={**self._headers(), "X-Tenant-ID": "tenant-b"},
            json={**self._body(key="spoofed"), "tenant_id": "tenant-b"},
        )

        self.assertEqual(403, missing.status_code)
        self.assertEqual("FORBIDDEN", missing.json()["error"]["code"])
        self.assertEqual(403, duplicate_auth.status_code)
        self.assertEqual(403, malformed_anonymous.status_code)
        self.assertEqual(422, spoofed.status_code)
        self.assertEqual(0, self.executor.execution_count)

        allowed = self.client.post(
            "/api/v1/runs",
            headers=self._headers(),
            json=self._body(key="authorized"),
        )
        run_id = allowed.json()["run_id"]
        cross_tenant = self.client.get(
            f"/api/v1/runs/{run_id}",
            headers=self._headers("token-b"),
        )
        self.authorizer.permissions.remove(ApiPermission.RUN_READ)
        revoked = self.client.get(
            f"/api/v1/runs/{run_id}",
            headers=self._headers(),
        )

        self.assertEqual(404, cross_tenant.status_code)
        self.assertEqual("NOT_FOUND", cross_tenant.json()["error"]["code"])
        self.assertEqual(403, revoked.status_code)
        self.assertEqual("FORBIDDEN", revoked.json()["error"]["code"])

        self.authorizer.permissions.add(ApiPermission.RUN_READ)
        self.authorizer.projects[("tenant-a", "Case-001")] = "project-other"
        moved_project = self.client.get(
            f"/api/v1/runs/{run_id}",
            headers=self._headers(),
        )
        self.assertEqual(403, moved_project.status_code)
        self.assertEqual("FORBIDDEN", moved_project.json()["error"]["code"])

    def test_raw_body_and_media_type_are_rejected_before_execution(self) -> None:
        small_client = self._client(max_request_body_bytes=256)
        oversized = small_client.post(
            "/api/v1/runs",
            headers={**self._headers(), "Content-Type": "application/json"},
            content=b"{" + b"x" * 256,
        )
        compressed = self.client.post(
            "/api/v1/runs",
            headers={
                **self._headers(),
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
            content=b"{}",
        )
        wrong_type = self.client.post(
            "/api/v1/runs",
            headers={**self._headers(), "Content-Type": "text/plain"},
            content=b"{}",
        )

        self.assertEqual(413, oversized.status_code)
        self.assertEqual("INVALID_REQUEST", oversized.json()["error"]["code"])
        self.assertEqual(415, compressed.status_code)
        self.assertEqual(415, wrong_type.status_code)
        self.assertEqual(0, self.executor.execution_count)

    def test_approval_cancel_resume_and_conflict_guards(self) -> None:
        waiting = self.client.post(
            "/api/v1/runs",
            headers=self._headers(),
            json=self._body(
                case_id="Case-Approval",
                key="approval-run",
                metadata={"requires_approval": True, "approval_id": "approval-1"},
                artifacts=False,
            ),
        ).json()
        run_id = waiting["run_id"]
        self.assertEqual("WAITING_APPROVAL", waiting["status"])

        wrong = self.client.post(
            f"/api/v1/runs/{run_id}/approvals/wrong",
            headers=self._headers(),
            json={"approved": True},
        )
        approved = self.client.post(
            f"/api/v1/runs/{run_id}/approvals/approval-1",
            headers=self._headers(),
            json={"approved": True, "comment": "reviewed"},
        )
        repeated = self.client.post(
            f"/api/v1/runs/{run_id}/approvals/approval-1",
            headers=self._headers(),
            json={"approved": True},
        )

        self.assertEqual(409, wrong.status_code)
        self.assertEqual(200, approved.status_code)
        self.assertEqual("SUCCEEDED", approved.json()["status"])
        self.assertEqual(409, repeated.status_code)

        cancellable = self.client.post(
            "/api/v1/runs",
            headers=self._headers(),
            json=self._body(
                case_id="Case-Approval",
                key="cancel-run",
                metadata={"requires_approval": True},
                artifacts=False,
            ),
        ).json()
        cancelled = self.client.post(
            f"/api/v1/runs/{cancellable['run_id']}/cancel",
            headers=self._headers(),
        )
        cancel_again = self.client.post(
            f"/api/v1/runs/{cancellable['run_id']}/cancel",
            headers=self._headers(),
        )
        self.assertEqual("CANCELLED", cancelled.json()["status"])
        self.assertEqual(409, cancel_again.status_code)

        failed = self.client.post(
            "/api/v1/runs",
            headers=self._headers(),
            json=self._body(
                case_id="Case-Failure",
                key="recover-run",
                metadata={"dependency_timeout": True},
                artifacts=False,
            ),
        ).json()
        resumed = self.client.post(
            f"/api/v1/runs/{failed['run_id']}/resume",
            headers=self._headers(),
        )
        self.assertEqual("FAILED", failed["status"])
        self.assertTrue(failed["checkpoint"]["recoverable"])
        self.assertEqual(200, resumed.status_code)
        self.assertEqual("FAILED", resumed.json()["status"])

    def test_fixed_validation_not_found_method_internal_and_health_errors(self) -> None:
        invalid = self.client.post(
            "/api/v1/runs",
            headers=self._headers(),
            json=self._body(case_id="../private", key="invalid"),
        )
        bad_path = self.client.get(
            "/api/v1/runs/bad%20id",
            headers=self._headers(),
        )
        not_found = self.client.get("/private-route")
        method = self.client.put("/health")
        down_client = self._client(health_probe=_FailingHealthProbe())
        down = down_client.get("/health")

        self.assertEqual(422, invalid.status_code)
        self.assertEqual("INVALID_REQUEST", invalid.json()["error"]["code"])
        self.assertNotIn("../private", invalid.text)
        self.assertEqual(400, bad_path.status_code)
        self.assertEqual("INVALID_ARGUMENT", bad_path.json()["error"]["code"])
        self.assertEqual(404, not_found.status_code)
        self.assertEqual("NOT_FOUND", not_found.json()["error"]["code"])
        self.assertEqual(405, method.status_code)
        self.assertEqual("INVALID_ARGUMENT", method.json()["error"]["code"])
        self.assertEqual(503, down.status_code)
        self.assertEqual({"status": "DOWN"}, down.json())
        self.assertNotIn("secret", down.text)

        class BrokenService:
            def create_run(self, request):
                del request
                raise RuntimeError("Authorization Bearer private patient-123")

        broken_client = self._client(run_service=BrokenService())
        broken = broken_client.post(
            "/api/v1/runs",
            headers=self._headers(),
            json=self._body(key="broken"),
        )
        self.assertEqual(500, broken.status_code)
        self.assertEqual("INTERNAL_ERROR", broken.json()["error"]["code"])
        self.assertNotIn("private", broken.text)
        self.assertNotIn("patient-123", broken.text)

        strict_app = create_app(
            BrokenService(),
            authenticator=self.authenticator,
            authorizer=self.authorizer,
            artifact_gateway=self.artifacts,
        )
        with TestClient(strict_app) as strict_client:
            strict = strict_client.post(
                "/api/v1/runs",
                headers=self._headers(),
                json=self._body(key="strict-broken"),
            )
        self.assertEqual(500, strict.status_code)
        self.assertEqual("INTERNAL_ERROR", strict.json()["error"]["code"])

    def test_artifact_gateway_is_explicit_and_unconfigured_gateway_is_retryable(self) -> None:
        client = self._client(artifact_gateway=None)
        response = client.get(
            "/api/v1/artifacts/artifact-001/metadata",
            headers=self._headers(),
        )

        self.assertEqual(503, response.status_code)
        self.assertEqual("SERVICE_UNAVAILABLE", response.json()["error"]["code"])
        self.assertTrue(response.json()["error"]["retryable"])
        health = client.get("/health")
        self.assertEqual({"status": "DEGRADED"}, health.json())
        self.assertEqual("no-store", health.headers["cache-control"])
        self.assertEqual("nosniff", health.headers["x-content-type-options"])

    def test_artifact_metadata_rejects_wrong_identity_sensitive_fields_and_acl(self) -> None:
        original = self.artifacts.records["artifact-001"]
        self.artifacts.records["artifact-001"] = ArtifactPublicView(
            artifact_id="artifact-other",
            case_id="Case-001",
            artifact_type=ArtifactType.CT_VOLUME,
            status=ArtifactStatus.AVAILABLE,
            producer_name="case-loader",
            producer_version="1.0",
            geometry_fingerprint="geometry-public",
        )
        wrong_id = self.client.get(
            "/api/v1/artifacts/artifact-001/metadata",
            headers=self._headers(),
        )
        self.assertEqual(500, wrong_id.status_code)
        self.assertEqual("INTERNAL_ERROR", wrong_id.json()["error"]["code"])

        self.artifacts.records["artifact-001"] = ArtifactPublicView(
            artifact_id="artifact-001",
            case_id="Case-001",
            artifact_type=ArtifactType.CT_VOLUME,
            status=ArtifactStatus.AVAILABLE,
            producer_name="Bearer private-token",
            producer_version="1.0",
            geometry_fingerprint="geometry-public",
        )
        sensitive = self.client.get(
            "/api/v1/artifacts/artifact-001/metadata",
            headers=self._headers(),
        )
        self.assertEqual(500, sensitive.status_code)
        self.assertNotIn("private-token", sensitive.text)

        self.artifacts.records["artifact-001"] = original
        self.artifacts.denied_tenants.add("tenant-b")
        denied = self.client.get(
            "/api/v1/artifacts/artifact-001/metadata",
            headers=self._headers("token-b"),
        )
        self.assertEqual(403, denied.status_code)
        self.assertEqual("FORBIDDEN", denied.json()["error"]["code"])

        bad_path = self.client.get(
            "/api/v1/artifacts/artifact@1/metadata",
            headers=self._headers(),
        )
        self.assertEqual(400, bad_path.status_code)
        self.assertEqual("INVALID_ARGUMENT", bad_path.json()["error"]["code"])

    def test_concurrent_idempotency_and_cross_tenant_scope(self) -> None:
        body = self._body(key="concurrent-create")

        def create(token: str) -> tuple[int, str]:
            response = self.client.post(
                "/api/v1/runs",
                headers=self._headers(token),
                json=body,
            )
            return response.status_code, response.json()["run_id"]

        with ThreadPoolExecutor(max_workers=12) as pool:
            tenant_a = list(pool.map(lambda _: create("token-a"), range(30)))
        tenant_b = create("token-b")

        self.assertEqual({200}, {status for status, _ in tenant_a})
        self.assertEqual(1, len({run_id for _, run_id in tenant_a}))
        self.assertEqual(200, tenant_b[0])
        self.assertNotEqual(tenant_a[0][1], tenant_b[1])
        self.assertEqual(2, self.executor.execution_count)


def _psycopg_available() -> bool:
    return importlib.util.find_spec("psycopg") is not None


@unittest.skipUnless(
    FASTAPI_AVAILABLE and POSTGRES_DSN and _psycopg_available(),
    "FastAPI/PostgreSQL integration environment is not configured",
)
class FastApiPostgresIntegrationTests(unittest.TestCase):
    def test_postgres_factory_migrates_healthchecks_and_persists_idempotency(self) -> None:
        import psycopg

        from puncture_agent.api.postgres_app import (
            PostgresApiSettings,
            create_postgres_app,
        )

        schema = f"fastapi_{uuid4().hex[:20]}"
        authenticator = _Authenticator()
        authorizer = _Authorizer()
        artifacts = _ArtifactGateway()
        executor = ScenarioExecutor()
        settings = PostgresApiSettings(
            connection_string=POSTGRES_DSN,
            schema=schema,
            migrate_on_startup=True,
        )
        unmigrated_app = create_postgres_app(
            PostgresApiSettings(
                connection_string=POSTGRES_DSN,
                schema=schema,
                migrate_on_startup=False,
            ),
            executor=executor,
            authenticator=authenticator,
            authorizer=authorizer,
            artifact_gateway=artifacts,
        )
        with TestClient(unmigrated_app, raise_server_exceptions=False) as client:
            self.assertEqual({"status": "DOWN"}, client.get("/health").json())
        app = create_postgres_app(
            settings,
            executor=executor,
            authenticator=authenticator,
            authorizer=authorizer,
            artifact_gateway=artifacts,
        )
        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                body = FastApiTransportTests._body(key="postgres-http")
                first = client.post(
                    "/api/v1/runs",
                    headers=FastApiTransportTests._headers(),
                    json=body,
                )
                second = client.post(
                    "/api/v1/runs",
                    headers=FastApiTransportTests._headers(),
                    json=body,
                )
                health = client.get("/health")

                self.assertEqual(200, first.status_code, first.text)
                self.assertEqual(first.json()["run_id"], second.json()["run_id"])
                self.assertEqual(1, executor.execution_count)
                self.assertEqual({"status": "UP"}, health.json())
                with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
                    connection.execute(f'DROP TABLE "{schema}".run_events')
                self.assertEqual(
                    {"status": "DOWN"},
                    client.get("/health").json(),
                )
        finally:
            with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
                connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


if __name__ == "__main__":
    unittest.main()
