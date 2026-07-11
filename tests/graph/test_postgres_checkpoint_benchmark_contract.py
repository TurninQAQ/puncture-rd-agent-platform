"""Offline contract tests for the PostgreSQL checkpoint benchmark."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]

from benchmarks import langgraph_postgres_checkpoint_benchmark as benchmark


class PostgresCheckpointBenchmarkContractTests(unittest.TestCase):
    def test_nearest_rank_and_round_summary_are_reproducible(self) -> None:
        self.assertEqual(4.0, benchmark.nearest_rank([4, 1, 3, 2], 0.95))
        summary = benchmark.summarize_rounds([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(6, summary["count"])
        self.assertEqual(3.5, summary["p50"])
        self.assertEqual(6.0, summary["p95"])
        self.assertEqual([3.0, 6.0], summary["round_p95"])
        with self.assertRaises(ValueError):
            benchmark.nearest_rank([], 0.95)
        with self.assertRaises(ValueError):
            benchmark.nearest_rank([1], 0.0)

    def test_parser_rejects_invalid_counts_and_secret_like_labels(self) -> None:
        invalid_arguments = (
            ["--warmups", "-1"],
            ["--rounds", "0"],
            ["--samples-per-round", "0"],
            ["--environment-label", "postgresql://secret"],
            ["--storage-label", "user@database"],
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    benchmark.parse_args(arguments)

    def test_timing_recorder_filters_operations_by_phase(self) -> None:
        recorder = benchmark.TimingRecorder()
        with recorder.phase("round-0:prepare"):
            recorder.record("put", 1.25)
            recorder.record("put_writes", 0.5)
        with recorder.phase("round-0:resume"):
            recorder.record("put", 2.5)
        self.assertEqual(
            [1.25, 2.5],
            recorder.durations(
                "put",
                {"round-0:prepare", "round-0:resume"},
            ),
        )
        self.assertEqual(
            [0.5],
            recorder.durations("put_writes", {"round-0:prepare"}),
        )
        recorder.clear()
        self.assertEqual([], recorder.samples)

    def test_schema_pins_thresholds_without_connection_secrets(self) -> None:
        schema_path = (
            PROJECT_ROOT
            / "benchmarks"
            / "schemas"
            / "langgraph-postgres-checkpoint-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        threshold_properties = schema["properties"]["thresholds"]["properties"]
        self.assertEqual(50.0, threshold_properties["save_p95_ms"]["const"])
        self.assertEqual(150.0, threshold_properties["resume_p95_ms"]["const"])

        property_names: set[str] = set()

        def collect(value) -> None:
            if isinstance(value, dict):
                properties = value.get("properties")
                if isinstance(properties, dict):
                    property_names.update(str(name).lower() for name in properties)
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(schema)
        self.assertTrue(
            {"dsn", "password", "username", "hostname", "database"}.isdisjoint(
                property_names
            )
        )

        def fake_run_batch(*, recorder, label, sample_count, **_kwargs):
            with recorder.phase(f"{label}:prepare"):
                recorder.record("put", 1.0)
                recorder.record("put_writes", 0.5)
            with recorder.phase(f"{label}:resume"):
                recorder.record("put", 2.0)
                recorder.record("put_writes", 0.75)
            return benchmark.BatchMeasurements(
                checkpoint_state_bytes=[1024.0] * sample_count,
                resume_ms=[10.0] * sample_count,
                run_to_interrupt_ms=[20.0] * sample_count,
            )

        args = benchmark.parse_args(
            ["--warmups", "0", "--rounds", "1", "--samples-per-round", "1"]
        )
        postgres = {
            "server_version": "16.test",
            "server_version_num": 160000,
            "synchronous_commit": "on",
        }
        with (
            mock.patch.object(benchmark, "langgraph_available", return_value=True),
            mock.patch.object(benchmark, "run_batch", side_effect=fake_run_batch),
            mock.patch.object(
                benchmark,
                "postgres_environment",
                return_value=postgres,
            ),
            mock.patch.object(benchmark, "package_version", return_value="test"),
            mock.patch.object(benchmark, "cpu_model", return_value="test-cpu"),
            mock.patch.object(benchmark, "git_value", return_value=""),
            mock.patch.dict(os.environ, {"GITHUB_SHA": "a" * 40}, clear=False),
        ):
            result = benchmark.build_result(args, "redacted-test-dsn")

        self.assertEqual(set(schema["required"]), set(result))
        workload_schema = schema["properties"]["workload"]
        self.assertEqual(set(workload_schema["required"]), set(result["workload"]))
        self.assertEqual(
            set(schema["properties"]["measurements"]["required"]),
            set(result["measurements"]),
        )
        self.assertNotIn("redacted-test-dsn", json.dumps(result))

    def test_output_is_private_and_missing_dsn_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.json"
            benchmark.write_json(output, {"schema_version": "1"})
            self.assertEqual(
                stat.S_IRUSR | stat.S_IWUSR,
                stat.S_IMODE(output.stat().st_mode),
            )
            self.assertEqual(
                {"schema_version": "1"},
                json.loads(output.read_text(encoding="utf-8")),
            )

            missing_output = Path(temporary_directory) / "missing.json"
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                redirect_stderr(StringIO()) as stderr,
            ):
                exit_code = benchmark.main(
                    [
                        "--warmups",
                        "0",
                        "--rounds",
                        "1",
                        "--samples-per-round",
                        "1",
                        "--output",
                        str(missing_output),
                    ]
                )
            self.assertEqual(1, exit_code)
            self.assertIn(benchmark.DSN_ENV, stderr.getvalue())
            self.assertFalse(missing_output.exists())


if __name__ == "__main__":
    unittest.main()
