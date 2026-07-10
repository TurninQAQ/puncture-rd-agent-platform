"""API adapter package; the mock stage exposes the framework-neutral service."""

from puncture_agent.runtime import InMemoryRunService, RunRequest

__all__ = ["InMemoryRunService", "RunRequest"]
