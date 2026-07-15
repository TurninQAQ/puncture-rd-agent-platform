"""Composition adapter for the complete no-dependency mock workflow."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from puncture_agent.agent import (
    AgentState,
    AgentStatus,
    GraphRuntime,
    TaskType,
    build_mock_handlers,
)
from puncture_agent.agent.runtime import NodeOutcome
from puncture_agent.model_gateway import (
    ChatMessage,
    MockQwenGateway,
    ModelGateway,
    ModelRequest,
)
from puncture_agent.rag import MockRagService, RagService, RetrievalRequest

from .models import ApprovalDecision, EventType, ExecutionOutcome, RunRequest, RunStatus
from .service import Emit


class IntegratedMockExecutor:
    """Compose the real scaffold boundaries using deterministic doubles.

    It performs no medical computation and no real model inference.  Its purpose
    is to prove that the stable Qwen, RAG, graph, tool, verifier, API, and trace
    contracts can be connected before each production module is implemented.
    """

    def __init__(
        self,
        *,
        model_gateway: ModelGateway | None = None,
        rag_service: RagService | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self._graph_path = project_root / "graph" / "main_graph.json"
        self.model_gateway = model_gateway or MockQwenGateway()
        self.rag_service = rag_service or MockRagService.from_default_fixture()
        self.last_state: AgentState | None = None
        self.last_model_response: Any = None
        self.last_rag_response: Any = None

    def execute(
        self,
        request: RunRequest,
        emit: Emit,
        *,
        checkpoint: Mapping[str, Any] | None = None,
        approval: ApprovalDecision | None = None,
    ) -> ExecutionOutcome:
        if approval is not None and not approval.approved:
            return ExecutionOutcome(
                status=RunStatus.FAILED,
                checkpoint=dict(checkpoint or {}),
                error={"code": "APPROVAL_REJECTED", "message": approval.comment or "rejected"},
            )

        emit(EventType.NODE_STARTED, "model_gateway.plan", {"model": "mock-qwen-instruct"})
        model_response = self.model_gateway.generate(
            ModelRequest(
                request_id=f"plan-{request.idempotency_key}",
                messages=(
                    ChatMessage(
                        role="system",
                        content="Return a task classification and case ID; never calculate medical safety.",
                    ),
                    ChatMessage(role="user", content=request.user_query),
                ),
                response_schema={
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": ["DATA_MODEL_VALIDATION", "PLANNING_SAFETY"],
                        },
                        "case_id": {"type": "string"},
                    },
                    "required": ["task_type", "case_id"],
                    "additionalProperties": False,
                },
                metadata={
                    "mock_structured_output": {
                        "task_type": request.task_type,
                        "case_id": request.case_id,
                    }
                },
            )
        )
        self.last_model_response = model_response
        emit(
            EventType.NODE_COMPLETED,
            "model_gateway.plan",
            {
                "finish_reason": model_response.finish_reason,
                "total_tokens": model_response.usage.total_tokens,
            },
        )

        modules = (
            ("path_planning", "safety_evaluation")
            if request.task_type == TaskType.PLANNING_SAFETY
            else ("data_validation", "segmentation")
        )
        retrieval_query = (
            f"{request.user_query} needle path safety envelope skin penetration"
            if request.task_type == TaskType.PLANNING_SAFETY
            else f"{request.user_query} NIfTI label geometry spacing segmentation"
        )
        scopes = tuple(request.metadata.get("access_scopes", ("public", "algorithm_team")))
        emit(EventType.NODE_STARTED, "rag.retrieve", {"modules": list(modules)})
        rag_response = self.rag_service.retrieve(
            RetrievalRequest(
                request_id=f"rag-{request.idempotency_key}",
                query=retrieval_query,
                modules=modules,
                access_scopes=scopes,
                top_k=3,
            )
        )
        self.last_rag_response = rag_response
        emit(
            EventType.NODE_COMPLETED,
            "rag.retrieve",
            {"chunk_count": len(rag_response.chunks), "warnings": list(rag_response.warnings)},
        )

        state = AgentState(
            user_query=request.user_query,
            case_id=request.case_id,
            task_type=request.task_type,
            metadata={**dict(request.metadata), "model_plan": dict(model_response.structured_output or {})},
        )
        state.artifacts.update({f"input_{index}": value for index, value in enumerate(request.artifact_ids)})

        handlers = build_mock_handlers()

        def retrieve_project_knowledge(agent_state: AgentState, _: Any) -> NodeOutcome:
            agent_state.retrieved_documents = [asdict(chunk) for chunk in rag_response.chunks]
            agent_state.citations = [
                {
                    "document_id": chunk.document_id,
                    "version": chunk.version,
                    "section": chunk.section,
                    "citation": chunk.citation,
                }
                for chunk in rag_response.chunks
            ]
            return NodeOutcome(output={"document_count": len(rag_response.chunks)})

        handlers["retrieve_project_knowledge"] = retrieve_project_knowledge
        emit(EventType.NODE_STARTED, "agent.graph", {"graph": "puncture_agent_main"})
        runtime = GraphRuntime(self._graph_path, handlers)
        state = runtime.run(state)
        self.last_state = state
        emit(
            EventType.NODE_COMPLETED,
            "agent.graph",
            {
                "visited_node_count": len(state.visited_nodes),
                "tool_call_count": len(state.tool_calls),
                "verification_status": state.verification_status,
            },
        )

        checkpoint_payload = state.to_dict()
        if state.status in {
            AgentStatus.SUCCEEDED,
            AgentStatus.COMPLETED_WITH_NO_RESULT,
            AgentStatus.MANUAL_REVIEW,
        }:
            report = dict(state.final_report)
            report["runtime_evidence"] = {
                "model": model_response.model,
                "model_finish_reason": model_response.finish_reason,
                "rag_chunk_count": len(rag_response.chunks),
                "tool_call_count": len(state.tool_calls),
                "visited_node_count": len(state.visited_nodes),
            }
            return ExecutionOutcome(
                status=RunStatus.SUCCEEDED,
                final_report=report,
                checkpoint=checkpoint_payload,
            )

        return ExecutionOutcome(
            status=RunStatus.FAILED,
            final_report=dict(state.final_report),
            checkpoint=checkpoint_payload,
            error={
                "code": "AGENT_WORKFLOW_FAILED",
                "message": f"agent terminated with status {state.status}",
                "retryable": False,
            },
        )
