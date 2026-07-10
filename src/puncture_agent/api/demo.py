"""Run one no-dependency end-to-end API/runtime demonstration."""

from __future__ import annotations

import json
from dataclasses import asdict

from puncture_agent.runtime import InMemoryRunService, IntegratedMockExecutor, RunRequest


def main() -> None:
    service = InMemoryRunService(IntegratedMockExecutor())
    snapshot = service.create_run(
        RunRequest(
            case_id="case-demo-001",
            user_query="检查 Case-demo-001 的 MCS 标签、分割结果并生成可追踪报告",
            task_type="DATA_MODEL_VALIDATION",
            idempotency_key="demo-create-001",
            artifact_ids=("ct-demo-001", "mcs-demo-001"),
        )
    )
    events = service.get_events(snapshot.run_id, tenant_id="default")
    output = {
        "run_id": snapshot.run_id,
        "status": snapshot.status.value,
        "final_report": dict(snapshot.final_report),
        "event_count": len(events),
        "visited_event_types": [event.event_type.value for event in events],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
