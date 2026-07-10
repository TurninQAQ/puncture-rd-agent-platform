"""Contract-first mock runtime for the puncture R&D agent.

The package intentionally uses only the Python standard library.  A later
implementation can replace :class:`GraphRuntime` with LangGraph while keeping
the JSON graph specifications, :class:`AgentState`, node names, and tests.
"""

from .graph_spec import GraphSpec, GraphSpecError, load_graph_spec, validate_graph_spec
from .nodes import build_mock_handlers
from .runtime import GraphExecutionError, GraphRuntime
from .state import AgentState, AgentStatus, TaskType, VerificationStatus

__all__ = [
    "AgentState",
    "AgentStatus",
    "GraphExecutionError",
    "GraphRuntime",
    "GraphSpec",
    "GraphSpecError",
    "TaskType",
    "VerificationStatus",
    "build_mock_handlers",
    "load_graph_spec",
    "validate_graph_spec",
]
