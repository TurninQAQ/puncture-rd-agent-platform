"""Cross-runtime LangGraph execution-lease integration tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from typing import Any
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent import (  # noqa: E402
    AgentState,
    AgentStatus,
    LangGraphCheckpointError,
    LangGraphConcurrencyError,
    LangGraphLeaseLostError,
    LangGraphLeaseUnavailableError,
    LangGraphRuntime,
    SQLiteThreadExecutionLeaseManager,
    ThreadLeaseLost,
    ThreadLeaseUnavailable,
    build_mock_handlers,
)
from tests.graph.test_langgraph_runtime import (  # noqa: E402
    FakeInMemorySaver,
    FakeLangGraphApi,
)


def _runtime(
    handlers: dict[str, Any],
    saver: FakeInMemorySaver,
    manager: Any,
) -> LangGraphRuntime:
    return LangGraphRuntime(
        PROJECT_ROOT / "graph" / "main_graph.json",
        handlers,
        checkpointer=saver,
        execution_lease_manager=manager,
        langgraph_api=FakeLangGraphApi(),
    )


class _UnavailableManager:
    def __init__(self) -> None:
        self.acquire_calls = 0

    def acquire(self, thread_id: str, *, operation: str) -> Any:
        del thread_id, operation
        self.acquire_calls += 1
        raise ThreadLeaseUnavailable("simulated lease backend outage")


class _LossControlledLease:
    def __init__(self, thread_id: str, operation: str) -> None:
        self.thread_id = thread_id
        self.operation = operation
        self.owner_token = "controlled-owner"
        self.backend = "controlled-test"
        self.lost = False
        self.released = False
        self.renew_calls = 0
        self.assert_calls = 0

    def renew(self) -> None:
        self.renew_calls += 1
        if self.lost:
            raise ThreadLeaseLost("controlled lease was lost")

    def assert_valid(self) -> None:
        self.assert_calls += 1
        if self.lost:
            raise ThreadLeaseLost("controlled lease was lost")

    def release(self) -> None:
        self.released = True

    def __enter__(self) -> "_LossControlledLease":
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


class _LossControlledManager:
    def __init__(self) -> None:
        self.acquire_calls = 0
        self.lease: _LossControlledLease | None = None

    def acquire(self, thread_id: str, *, operation: str) -> Any:
        self.acquire_calls += 1
        self.lease = _LossControlledLease(thread_id, operation)
        return self.lease


class LangGraphThreadLeaseIntegrationTests(unittest.TestCase):
    def test_two_runtimes_reject_same_thread_before_second_handler(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "thread-leases.sqlite3"
            first_manager = SQLiteThreadExecutionLeaseManager(path)
            second_manager = SQLiteThreadExecutionLeaseManager(path)
            self.addCleanup(first_manager.close)
            self.addCleanup(second_manager.close)
            saver = FakeInMemorySaver()
            entered = Event()
            release = Event()
            first_handlers = dict(build_mock_handlers())
            first_parse = first_handlers["parse_request"]

            def blocking_parse(state: AgentState, context: Any) -> Any:
                entered.set()
                if not release.wait(timeout=3.0):
                    raise RuntimeError("test synchronization timeout")
                return first_parse(state, context)

            first_handlers["parse_request"] = blocking_parse
            second_calls: list[str] = []
            second_handlers = dict(build_mock_handlers())
            second_parse = second_handlers["parse_request"]

            def counted_parse(state: AgentState, context: Any) -> Any:
                second_calls.append("parse_request")
                return second_parse(state, context)

            second_handlers["parse_request"] = counted_parse
            first_runtime = _runtime(first_handlers, saver, first_manager)
            second_runtime = _runtime(second_handlers, saver, second_manager)
            first_state = AgentState(
                user_query="对 Case-811 做路径规划",
                session_id="cross-runtime-same-thread",
            )
            second_state = AgentState(
                user_query="对 Case-811 做路径规划",
                session_id="cross-runtime-same-thread",
            )
            second_before = deepcopy(second_state.to_dict())

            with ThreadPoolExecutor(max_workers=2) as pool:
                first_future = pool.submit(first_runtime.run, first_state)
                self.assertTrue(entered.wait(timeout=2.0))
                puts_before = saver.put_count
                with self.assertRaises(LangGraphConcurrencyError):
                    second_runtime.run(second_state)
                self.assertEqual(puts_before, saver.put_count)
                self.assertEqual(second_before, second_state.to_dict())
                self.assertEqual([], second_calls)
                release.set()
                completed = first_future.result(timeout=3.0)

            self.assertEqual(AgentStatus.SUCCEEDED, completed.status)

    def test_different_threads_enter_handlers_concurrently(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "thread-leases.sqlite3"
            first_manager = SQLiteThreadExecutionLeaseManager(path)
            second_manager = SQLiteThreadExecutionLeaseManager(path)
            self.addCleanup(first_manager.close)
            self.addCleanup(second_manager.close)
            saver = FakeInMemorySaver()
            both_entered = Event()
            release = Event()
            count_lock = Lock()
            entered_count = 0

            def handlers() -> dict[str, Any]:
                nonlocal entered_count
                result = dict(build_mock_handlers())
                original = result["parse_request"]

                def blocking_parse(state: AgentState, context: Any) -> Any:
                    nonlocal entered_count
                    with count_lock:
                        entered_count += 1
                        if entered_count == 2:
                            both_entered.set()
                    if not release.wait(timeout=3.0):
                        raise RuntimeError("test synchronization timeout")
                    return original(state, context)

                result["parse_request"] = blocking_parse
                return result

            first_runtime = _runtime(handlers(), saver, first_manager)
            second_runtime = _runtime(handlers(), saver, second_manager)
            first_state = AgentState(
                user_query="对 Case-812 做路径规划",
                session_id="cross-runtime-thread-a",
            )
            second_state = AgentState(
                user_query="对 Case-813 做路径规划",
                session_id="cross-runtime-thread-b",
            )

            with ThreadPoolExecutor(max_workers=2) as pool:
                first_future = pool.submit(first_runtime.run, first_state)
                second_future = pool.submit(second_runtime.run, second_state)
                self.assertTrue(both_entered.wait(timeout=2.0))
                release.set()
                first_result = first_future.result(timeout=3.0)
                second_result = second_future.result(timeout=3.0)

            self.assertEqual(AgentStatus.SUCCEEDED, first_result.status)
            self.assertEqual(AgentStatus.SUCCEEDED, second_result.status)

    def test_stream_holds_lease_until_drain_and_blocks_run(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "thread-leases.sqlite3"
            stream_manager = SQLiteThreadExecutionLeaseManager(path)
            run_manager = SQLiteThreadExecutionLeaseManager(path)
            self.addCleanup(stream_manager.close)
            self.addCleanup(run_manager.close)
            saver = FakeInMemorySaver()
            entered = Event()
            release = Event()
            stream_handlers = dict(build_mock_handlers())
            original = stream_handlers["parse_request"]

            def blocking_parse(state: AgentState, context: Any) -> Any:
                entered.set()
                if not release.wait(timeout=3.0):
                    raise RuntimeError("test synchronization timeout")
                return original(state, context)

            stream_handlers["parse_request"] = blocking_parse
            run_calls: list[str] = []
            run_handlers = dict(build_mock_handlers())
            run_original = run_handlers["parse_request"]

            def counted_parse(state: AgentState, context: Any) -> Any:
                run_calls.append("parse_request")
                return run_original(state, context)

            run_handlers["parse_request"] = counted_parse
            stream_runtime = _runtime(stream_handlers, saver, stream_manager)
            run_runtime = _runtime(run_handlers, saver, run_manager)
            streamed = AgentState(
                user_query="对 Case-816 做路径规划",
                session_id="cross-runtime-stream-run",
            )
            rejected = AgentState(
                user_query="对 Case-816 做路径规划",
                session_id="cross-runtime-stream-run",
            )
            rejected_before = deepcopy(rejected.to_dict())

            with ThreadPoolExecutor(max_workers=2) as pool:
                stream_future = pool.submit(
                    lambda: tuple(stream_runtime.stream(streamed))
                )
                self.assertTrue(entered.wait(timeout=2.0))
                with self.assertRaises(LangGraphConcurrencyError):
                    run_runtime.run(rejected)
                self.assertEqual([], run_calls)
                self.assertEqual(rejected_before, rejected.to_dict())
                release.set()
                events = stream_future.result(timeout=3.0)

            self.assertGreater(len(events), 1)
            self.assertEqual("RUN_COMPLETED", events[-1].event_type)

    def test_two_runtimes_reject_concurrent_resume_before_checkpoint_read(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "thread-leases.sqlite3"
            first_manager = SQLiteThreadExecutionLeaseManager(path)
            second_manager = SQLiteThreadExecutionLeaseManager(path)
            self.addCleanup(first_manager.close)
            self.addCleanup(second_manager.close)
            saver = FakeInMemorySaver()
            session_id = "cross-runtime-resume-resume"
            seed_runtime = _runtime(
                dict(build_mock_handlers()),
                saver,
                first_manager,
            )
            waiting = seed_runtime.run(
                AgentState(
                    user_query="执行路径规划和安全评估",
                    session_id=session_id,
                )
            )
            self.assertEqual(AgentStatus.AWAITING_INPUT, waiting.status)

            entered = Event()
            release = Event()
            first_handlers = dict(build_mock_handlers())
            original = first_handlers["parse_request"]

            def blocking_parse(state: AgentState, context: Any) -> Any:
                entered.set()
                if not release.wait(timeout=3.0):
                    raise RuntimeError("test synchronization timeout")
                return original(state, context)

            first_handlers["parse_request"] = blocking_parse
            second_calls: list[str] = []
            second_handlers = dict(build_mock_handlers())
            second_original = second_handlers["parse_request"]

            def counted_parse(state: AgentState, context: Any) -> Any:
                second_calls.append("parse_request")
                return second_original(state, context)

            second_handlers["parse_request"] = counted_parse
            first_runtime = _runtime(first_handlers, saver, first_manager)
            second_runtime = _runtime(second_handlers, saver, second_manager)

            with ThreadPoolExecutor(max_workers=2) as pool:
                first_future = pool.submit(
                    first_runtime.resume,
                    thread_id=session_id,
                    updates={"case_id": "Case-817"},
                )
                self.assertTrue(entered.wait(timeout=2.0))
                puts_before = saver.put_count
                with self.assertRaises(LangGraphConcurrencyError):
                    second_runtime.resume(
                        thread_id=session_id,
                        updates={"case_id": "Case-818"},
                    )
                self.assertEqual(puts_before, saver.put_count)
                self.assertEqual([], second_calls)
                release.set()
                resumed = first_future.result(timeout=3.0)

            self.assertEqual(AgentStatus.SUCCEEDED, resumed.status)
            self.assertEqual("Case-817", resumed.case_id)

    def test_unavailable_backend_fails_closed_without_state_or_checkpoint(self) -> None:
        saver = FakeInMemorySaver()
        manager = _UnavailableManager()
        calls: list[str] = []
        handlers = dict(build_mock_handlers())
        original = handlers["parse_request"]

        def counted_parse(state: AgentState, context: Any) -> Any:
            calls.append("parse_request")
            return original(state, context)

        handlers["parse_request"] = counted_parse
        runtime = _runtime(handlers, saver, manager)
        state = AgentState(
            user_query="对 Case-814 做路径规划",
            session_id="lease-backend-unavailable",
        )
        before = deepcopy(state.to_dict())

        with self.assertRaises(LangGraphLeaseUnavailableError):
            runtime.run(state)

        self.assertEqual(1, manager.acquire_calls)
        self.assertEqual([], calls)
        self.assertEqual(0, saver.put_count)
        self.assertEqual(before, state.to_dict())

    def test_loss_after_handler_stops_checkpoint_and_poison_thread(self) -> None:
        saver = FakeInMemorySaver()
        manager = _LossControlledManager()
        calls: list[str] = []
        handlers = dict(build_mock_handlers())
        for handler_name, original in tuple(handlers.items()):

            def tracked(
                state: AgentState,
                context: Any,
                *,
                _name: str = handler_name,
                _original: Any = original,
            ) -> Any:
                calls.append(_name)
                outcome = _original(state, context)
                if _name == "parse_request":
                    assert manager.lease is not None
                    state.session_id = "forged-other-thread"
                    manager.lease.lost = True
                return outcome

            handlers[handler_name] = tracked

        runtime = _runtime(handlers, saver, manager)
        state = AgentState(
            user_query="对 Case-815 做路径规划",
            session_id="lease-lost-after-handler",
        )

        with self.assertRaises(LangGraphLeaseLostError) as captured:
            runtime.run(state)

        terminal = captured.exception.state
        self.assertIsNotNone(terminal)
        assert terminal is not None
        self.assertEqual(AgentStatus.MANUAL_REVIEW, terminal.status)
        self.assertEqual("lease-lost-after-handler", terminal.session_id)
        self.assertEqual(terminal.to_dict(), state.to_dict())
        self.assertTrue(terminal.metadata["execution_state_uncertain"])
        self.assertTrue(terminal.metadata["execution_lease_lost"])
        lease_error = next(
            item for item in terminal.errors if item["code"] == "EXECUTION_LEASE_LOST"
        )
        self.assertFalse(lease_error["retryable"])
        self.assertEqual(["parse_request"], calls)
        self.assertEqual(0, saver.put_count)
        self.assertIsNotNone(manager.lease)
        assert manager.lease is not None
        self.assertTrue(manager.lease.released)

        retry = AgentState(
            user_query="对 Case-815 做路径规划",
            session_id=state.session_id,
        )
        with self.assertRaisesRegex(
            LangGraphCheckpointError,
            "manual checkpoint reconciliation",
        ):
            runtime.run(retry)
        self.assertEqual(AgentStatus.CREATED, retry.status)
        self.assertEqual(1, manager.acquire_calls)


if __name__ == "__main__":
    unittest.main()
