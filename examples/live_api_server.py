#!/usr/bin/env python3
"""Serve the explicit local full-stack API demo on a loopback address only."""

from __future__ import annotations

from dataclasses import replace
import hmac
import ipaddress
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


_DEMO_FLAG = "RUN_FULL_STACK_DEMO"
_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,256}$")


def require_demo_opt_in() -> None:
    if os.environ.get(_DEMO_FLAG) != "1":
        raise ValueError(f"set {_DEMO_FLAG}=1 to enable the local full-stack demo")


def _token_path() -> Path:
    configured = os.environ.get("PUNCTURE_DEMO_TOKEN_FILE", "var/local-demo/bearer-token")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _read_private_token(path: Path) -> str:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("local demo token path must be a regular non-symlink file")
    if metadata.st_nlink != 1 or metadata.st_mode & 0o077:
        raise ValueError("local demo token file must have mode 0600 and one hard link")
    raw = path.read_bytes()
    if len(raw) > 258 or not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError("local demo token file must contain exactly one bounded line")
    try:
        token = raw[:-1].decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("local demo token must be ASCII") from exc
    if not _TOKEN_PATTERN.fullmatch(token):
        raise ValueError("local demo token must be a strong URL-safe value")
    return token


def load_or_create_demo_token(path: Path | None = None) -> str:
    token_path = path or _token_path()
    token_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_mode = token_path.parent.stat().st_mode
    if not stat.S_ISDIR(parent_mode) or parent_mode & 0o077:
        raise ValueError("local demo token directory must not be group/world accessible")
    try:
        return _read_private_token(token_path)
    except FileNotFoundError:
        token = secrets.token_urlsafe(48)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(token_path, flags, 0o600)
        except FileExistsError:
            return _read_private_token(token_path)
        try:
            payload = token.encode("ascii") + b"\n"
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("local demo token write did not make progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return _read_private_token(token_path)


def _loopback_host(value: str) -> str:
    host = value.strip()
    if host.lower() == "localhost":
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    raise ValueError("PUNCTURE_DEMO_HOST must be a loopback address")


def _port(value: str) -> int:
    if not value.isdigit() or not 1 <= int(value) <= 65535:
        raise ValueError("PUNCTURE_DEMO_PORT must be an integer in [1, 65535]")
    return int(value)


def _allowed_cases() -> frozenset[str]:
    values = tuple(
        item.strip()
        for item in os.environ.get("PUNCTURE_DEMO_CASE_IDS", "Case-102,Case-203").split(",")
        if item.strip()
    )
    if not values or len(values) != len(set(values)):
        raise ValueError("PUNCTURE_DEMO_CASE_IDS must contain unique case IDs")
    return frozenset(values)


def _build_live_rag() -> Any:
    from examples.live_opensearch_rag_demo import (
        build_seed_documents,
        endpoint_from_environment,
        seed_documents,
    )
    from puncture_agent.rag import (
        DeterministicEmbeddingBackend,
        DeterministicReranker,
        EnterpriseRagClient,
        EnterpriseRagConfig,
        OpenSearchSearchBackend,
        RagDependencies,
        RagRuntimeConfig,
    )

    endpoint = endpoint_from_environment()
    read_alias = os.environ.get("RAG_READ_ALIAS", "puncture-rag-read")
    write_alias = os.environ.get("RAG_WRITE_ALIAS", "puncture-rag-write")
    search = OpenSearchSearchBackend(endpoint, read_alias=read_alias)
    try:
        descriptor = search.descriptor()
        embedding = DeterministicEmbeddingBackend(
            model_name=descriptor.embedding_model,
            revision=descriptor.embedding_revision,
            dimension=descriptor.embedding_dimension,
        )
        seed_documents(endpoint, write_alias, build_seed_documents(embedding))
        reranker = DeterministicReranker(
            model_name="deterministic-overlap",
            revision="1",
        )
        config = EnterpriseRagConfig(
            endpoint=endpoint.base_url,
            index_name=read_alias,
            embedding_model=embedding.model_name,
            reranker_model=reranker.model_name,
            timeout_seconds=15,
            dense_top_k=8,
            lexical_top_k=8,
            rerank_top_k=6,
        )
        return EnterpriseRagClient(
            config,
            dependencies=RagDependencies(search, embedding, reranker),
            runtime=RagRuntimeConfig(
                minimum_relevance=0.03,
                recall_mode="hybrid",
                use_reranker=True,
                expand_parent_context=False,
            ),
        )
    except BaseException:
        search.close()
        raise


def build_local_demo_app() -> Any:
    """Compose real FastAPI/PostgreSQL/vLLM/OpenSearch with synthetic tools."""

    require_demo_opt_in()

    from puncture_agent.api.fastapi_app import ApiPermission, AuthorizedCase
    from puncture_agent.api.http_contracts import AuthenticatedPrincipal
    from puncture_agent.api.postgres_app import PostgresApiSettings, create_postgres_app
    from puncture_agent.model_gateway import VllmGatewayConfig, VllmModelGateway
    from puncture_agent.runtime import IntegratedMockExecutor
    from puncture_agent.runtime.errors import RunServiceError

    token = load_or_create_demo_token()
    allowed_cases = _allowed_cases()
    principal = AuthenticatedPrincipal(
        "local-demo-tenant",
        "local-demo-user",
        ("team-a", "algorithm-team"),
    )
    for case_id in allowed_cases:
        AuthorizedCase(
            tenant_id=principal.tenant_id,
            project_id="local-demo-project",
            case_id=case_id,
        )

    class LocalDemoAuthenticator:
        def authenticate(self, bearer_token: str) -> AuthenticatedPrincipal:
            if not hmac.compare_digest(bearer_token, token):
                raise RunServiceError("FORBIDDEN", "invalid local demo token")
            return principal

    class LocalDemoAuthorizer:
        @staticmethod
        def _authorize(case_id: str, tenant_id: str) -> AuthorizedCase:
            if tenant_id != principal.tenant_id or case_id not in allowed_cases:
                raise RunServiceError("FORBIDDEN", "local demo case access denied")
            return AuthorizedCase(
                tenant_id=principal.tenant_id,
                project_id="local-demo-project",
                case_id=case_id,
            )

        def require_case(
            self,
            authenticated: AuthenticatedPrincipal,
            *,
            case_id: str,
            permission: ApiPermission,
        ) -> AuthorizedCase:
            del permission
            return self._authorize(case_id, authenticated.tenant_id)

        def require_run(
            self,
            authenticated: AuthenticatedPrincipal,
            *,
            snapshot: Any,
            permission: ApiPermission,
        ) -> AuthorizedCase:
            del permission
            if snapshot.request.tenant_id != authenticated.tenant_id:
                raise RunServiceError("FORBIDDEN", "local demo run access denied")
            return self._authorize(snapshot.request.case_id, authenticated.tenant_id)

    rag = _build_live_rag()
    gateway = VllmModelGateway(
        VllmGatewayConfig(
            base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8008/v1"),
            model=os.environ.get("VLLM_MODEL", "qwen-enterprise-agent"),
            timeout_seconds=float(os.environ.get("VLLM_TIMEOUT_SECONDS", "120")),
            max_retries=1,
        )
    )
    try:
        model_health = gateway.health()
        rag_health = rag.health()
        if model_health.status != "UP" or rag_health.status == "DOWN":
            raise RuntimeError("local demo model or RAG dependency is unavailable")

        class LocalDemoHealth:
            def status(self) -> str:
                current_model = gateway.health()
                current_rag = rag.health()
                if current_model.status == "DOWN" or current_rag.status == "DOWN":
                    raise RuntimeError("local demo dependency is down")
                if current_model.status == "DEGRADED" or current_rag.status == "DEGRADED":
                    return "DEGRADED"
                return "UP"

        source = dict(os.environ)
        source.setdefault("PUNCTURE_API_POSTGRES_SCHEMA", "puncture_local_demo")
        settings = replace(
            PostgresApiSettings.from_env(source),
            migrate_on_startup=True,
            worker_enabled=False,
        )
        executor = IntegratedMockExecutor(
            model_gateway=gateway,
            rag_service=rag,
        )
        app = create_postgres_app(
            settings,
            executor=executor,
            authenticator=LocalDemoAuthenticator(),
            authorizer=LocalDemoAuthorizer(),
            optional_health_probe=LocalDemoHealth(),
            additional_shutdown_hooks=(gateway.close, rag.close),
        )
    except BaseException:
        gateway.close()
        rag.close()
        raise

    app.state.local_demo = {
        "mode": "synthetic-tools",
        "model": gateway.config.model,
        "rag_backend": rag.health().backend,
        "case_count": len(allowed_cases),
    }
    return app


def main() -> int:
    try:
        require_demo_opt_in()
        host = _loopback_host(os.environ.get("PUNCTURE_DEMO_HOST", "127.0.0.1"))
        port = _port(os.environ.get("PUNCTURE_DEMO_PORT", "8010"))
        app = build_local_demo_app()
        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)
        return 0
    except Exception as exc:
        print(f"LIVE_API_SERVER_FAILED {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
