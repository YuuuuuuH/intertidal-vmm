from __future__ import annotations

import csv
import json
import struct
import subprocess
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

import gemma_capacity_sweep as base
import gemma_q8_context_sweep as q8


def checksum_log(labels: list[str], mutate: int | None = None) -> str:
    lines = []
    for index, label in enumerate(labels):
        value = index + 1 + (999 if index == mutate else 0)
        lines.append(f"logit_checksum,{label},{value:016x},nonfinite=0")
    return "\n".join(lines) + "\n"


def successful_result(context: int, generation: int, repetitions: int) -> base.CommandResult:
    prompt = context - generation
    record = [{"n_ctx": context, "n_prompt": prompt, "n_gen": generation, "n_depth": 0}]
    timings = []
    for rep in range(1, repetitions + 1):
        timings.extend(
            [
                f"llama_bench_phase_timing,phase=prompt,rep={rep},tokens={prompt},ns={prompt * 500_000}",
                f"llama_bench_phase_timing,phase=decode,rep={rep},tokens={generation},ns={generation * 10_000_000}",
            ]
        )
    stderr = "\n".join(timings) + "\n" + checksum_log(
        [label for _ in range(repetitions) for label in ("prompt", "decode")]
    )
    return base.CommandResult(0, json.dumps(record), stderr, 1.0)


def tiny_gguf() -> bytes:
    def gguf_string(value: bytes) -> bytes:
        return struct.pack("<Q", len(value)) + value

    name = b"blk.0.attn_q.weight"
    metadata = (
        gguf_string(b"general.architecture")
        + struct.pack("<I", 8)
        + gguf_string(b"gemma4")
        + gguf_string(b"gemma4.context_length")
        + struct.pack("<II", 4, 262144)
    )
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
    tensor = gguf_string(name) + struct.pack("<IQQIQ", 2, 32, 1, 8, 0)
    prefix = header + metadata + tensor
    return prefix + b"\0" * ((32 - len(prefix) % 32) % 32) + b"\0" * 34


class Q8ContextSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).with_name("q8-context-sweep-manifest.rtx5090.json")
        cls.manifest_path = path
        cls.manifest = json.loads(path.read_text())
        cls.model = q8.ModelInfo(
            31_300_000_000,
            30_600_000_000,
            1000,
            262144,
            "a" * 64,
            "b" * 64,
            123,
        )

    def test_manifest_summary_has_required_order_and_measurement(self) -> None:
        q8.validate_manifest(self.manifest)
        summary = q8.plan_summary(self.manifest, self.model)
        self.assertEqual(
            summary["execution_order"],
            [
                "wait_for_stable_download_and_parse_gguf_metadata",
                q8.PHASE_DIRECT,
                q8.PHASE_STAGING_COARSE,
                f"{q8.PHASE_ALTERNATES}:cuda_zero_copy",
                f"{q8.PHASE_ALTERNATES}:native_cpu_layers",
                q8.PHASE_FINE,
            ],
        )
        self.assertIn("-pg P,G", summary["populated_context"]["mechanism"])
        self.assertFalse(summary["populated_context"]["host_state_snapshot"])
        self.assertIn("llama_bench_phase_timing", summary["speed_sources"]["prefill"])
        self.assertTrue(summary["speed_sources"]["llama_perf_context_print_is_not_used"])
        self.assertTrue(summary["speed_sources"]["combined_json_avg_ts_is_not_used"])
        self.assertFalse(summary["remote_files_created"])

    def test_download_probe_parse_and_stability_gate_requires_metadata(self) -> None:
        manifest = json.loads(json.dumps(self.manifest))
        manifest["download"]["stable_polls"] = 2
        manifest["download"]["poll_interval_s"] = 0.001
        records = [
            f"{q8.MODEL_PREFIX}1|31300000000|100|1|1|1000|30600000000|262144|{'a' * 64}\n",
            f"{q8.MODEL_PREFIX}1|31300000000|101|0|0|0|0|0|-\n",
            f"{q8.MODEL_PREFIX}1|31300000000|102|0|1|1000|30600000000|262144|{'a' * 64}\n",
            f"{q8.MODEL_PREFIX}1|31300000000|102|0|1|1000|30600000000|262144|{'a' * 64}\n",
            f"{q8.MODEL_HASH_PREFIX}{'b' * 64}\n",
        ]
        results = [base.CommandResult(0, record, "", 0.01) for record in records]
        with mock.patch.object(base, "execute_wrapper", side_effect=results), mock.patch.object(
            q8.time, "sleep"
        ):
            info = q8.wait_for_model(manifest, dry_run=False, skip_wait=False)
        self.assertEqual(info, self.model.__class__(
            31_300_000_000, 30_600_000_000, 1000, 262144,
            "a" * 64, "b" * 64, 102,
        ))
        wrapper = q8.model_probe_wrapper("/tmp/model with spaces.gguf")
        self.assertIn(".aria2", wrapper)
        self.assertIn("GGUF", wrapper)
        self.assertNotIn("mktemp", wrapper)

    def test_embedded_gguf_reader_derives_eligible_q8_bytes(self) -> None:
        payload = tiny_gguf()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            path.write_bytes(payload)
            result = subprocess.run(
                ["python3", "-c", q8.GGUF_METADATA_PROBE, str(path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
        fields = result.stdout.strip().split("|")
        self.assertEqual(fields[:4], ["1", "1", "34", "262144"])
        self.assertRegex(fields[4], r"^[0-9a-f]{64}$")

    def test_complete_model_gate_wrapper_is_stdout_only_and_parseable(self) -> None:
        payload = tiny_gguf()
        local_manifest = json.loads(json.dumps(self.manifest))
        local_manifest["remote"] = {"transport": "local", "gpu_id": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny gguf.gguf"
            path.write_bytes(payload)
            result = base.execute_wrapper(
                local_manifest,
                q8.model_probe_wrapper(str(path)),
                timeout_s=10,
                show_telemetry=False,
            )
            self.assertEqual(list(Path(directory).iterdir()), [path])
        exists, info, partial, metadata_ok = q8.parse_model_probe(result.stdout)
        self.assertTrue(exists)
        self.assertFalse(partial)
        self.assertTrue(metadata_ok)
        self.assertEqual(info.eligible_tensor_bytes, 34)

    def test_percentage_uses_eligible_tensor_bytes(self) -> None:
        pages = q8.page_budget_for_pp(self.manifest, self.model, Decimal("0.1"))
        expected = round(30_600_000_000 * 0.001 / 2_097_152)
        self.assertEqual(pages, expected)

    def test_command_is_one_real_pg_context_without_depth_or_defaults(self) -> None:
        case = q8.build_case(
            self.manifest,
            self.model,
            q8.DMA,
            q8.PHASE_STAGING_COARSE,
            q8.SCREEN_PROFILE,
            "frontier",
            8192,
            target_pp=Decimal("6.0"),
        )
        env, command = q8.command_for_case(self.manifest, case)
        self.assertEqual(env["GGML_CUDA_HYBRID_MODE"], "staging")
        self.assertEqual(command[command.index("-p") + 1], "0")
        self.assertEqual(command[command.index("-n") + 1], "0")
        self.assertEqual(command[command.index("-pg") + 1], "8176,16")
        self.assertEqual(command[command.index("-d") + 1], "0")
        self.assertEqual(command[command.index("-nkvo") + 1], "0")
        self.assertEqual(command[command.index("--ctx-size") + 1], "8192")
        self.assertIn("--no-warmup", command)
        self.assertIn("-v", command)
        self.assertEqual(command[command.index("-o") + 1], "json")

    def test_vmm_zero_page_is_distinct_from_stock_direct(self) -> None:
        hybrid = q8.build_case(
            self.manifest,
            self.model,
            q8.DMA,
            q8.PHASE_STAGING_COARSE,
            q8.SCREEN_PROFILE,
            "shallow",
            4096,
            target_pp=Decimal(0),
        )
        direct = q8.build_case(
            self.manifest,
            self.model,
            q8.DIRECT,
            q8.PHASE_DIRECT,
            q8.SCREEN_PROFILE,
            "shallow",
            4096,
            target_pp=Decimal(0),
        )
        hybrid_env, hybrid_command = q8.command_for_case(self.manifest, hybrid)
        direct_env, direct_command = q8.command_for_case(self.manifest, direct)
        self.assertEqual(hybrid_env["GGML_CUDA_HYBRID_PAGE_BUDGET"], "0")
        self.assertIn("-ot", hybrid_command)
        self.assertNotIn("GGML_CUDA_HYBRID_PAGE_BUDGET", direct_env)
        self.assertNotIn("-ot", direct_command)

    def test_perf_parser_aggregates_strict_per_rep_machine_timings(self) -> None:
        case = q8.build_case(
            self.manifest,
            self.model,
            q8.DMA,
            q8.PHASE_STAGING_COARSE,
            q8.SCREEN_PROFILE,
            "shallow",
            4096,
            target_pp=Decimal("6"),
        )
        result = successful_result(4096, 16, 1)
        perf = q8.parse_performance(self.manifest, case, result)
        self.assertAlmostEqual(perf.prefill_tok_s, 2000.0)
        self.assertAlmostEqual(perf.decode_tok_s, 100.0)
        self.assertEqual(perf.prompt_samples, 4080)
        self.assertEqual(perf.decode_samples, 16)

        broken = base.CommandResult(
            0,
            result.stdout,
            result.stderr.replace("tokens=4080", "tokens=4079", 1),
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "sequence mismatch"):
            q8.parse_performance(self.manifest, case, broken)

        malformed = base.CommandResult(
            0,
            result.stdout,
            result.stderr.replace("ns=2040000000", "ns=0", 1),
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "malformed phase timing line"):
            q8.parse_performance(self.manifest, case, malformed)

        legacy_inf_only = base.CommandResult(
            0,
            result.stdout,
            "llama_perf_context_print: prompt eval time = 0.00 ms / 4080 tokens "
            "(0.00 ms per token, inf tokens per second)\n",
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "sequence mismatch"):
            q8.parse_performance(self.manifest, case, legacy_inf_only)

    def test_checksums_alternate_prompt_decode_and_match_same_context(self) -> None:
        labels = q8.expected_checksum_labels(self.manifest, q8.CONFIRM_PROFILE)
        self.assertEqual(labels, ["prompt", "decode"] * 3)
        case = q8.build_case(
            self.manifest,
            self.model,
            q8.DMA,
            q8.PHASE_CONFIRM,
            q8.CONFIRM_PROFILE,
            "frontier_confirmation",
            8192,
            target_pp=Decimal("10"),
        )
        result = successful_result(8192, 64, 3)
        captured = q8.evaluate_correctness(self.manifest, case, result, "ok", None)
        self.assertEqual(captured.status, "finite_baseline")

        zero = q8.build_case(
            self.manifest,
            self.model,
            q8.ZERO_COPY,
            q8.PHASE_CONFIRM,
            q8.CONFIRM_PROFILE,
            "frontier_confirmation",
            8192,
            target_pp=Decimal("10"),
        )
        match = q8.evaluate_correctness(
            self.manifest, zero, result, "ok", (q8.DMA, captured.entries)
        )
        self.assertEqual(match.status, "match")
        mismatch_result = base.CommandResult(
            0, result.stdout, checksum_log(labels, mutate=3) + result.stderr.split("logit_checksum", 1)[0], 1.0
        )
        mismatch = q8.evaluate_correctness(
            self.manifest, zero, mismatch_result, "ok", (q8.DMA, captured.entries)
        )
        self.assertEqual(mismatch.status, "fail")

    def test_direct_oom_is_expected_and_not_a_fake_baseline(self) -> None:
        case = q8.build_case(
            self.manifest,
            self.model,
            q8.DIRECT,
            q8.PHASE_DIRECT,
            q8.SCREEN_PROFILE,
            "shallow",
            4096,
            target_pp=Decimal(0),
        )
        result = base.CommandResult(1, "", "CUDA error: out of memory", 1.0)
        status, perf, error = q8.classify_result(self.manifest, case, result, self.model)
        row = q8.row_for_result(
            self.manifest,
            "hash",
            "run",
            self.model,
            case,
            {},
            ["bench"],
            result,
            status,
            perf,
            error,
            q8.CorrectnessEvaluation("not_run"),
            None,
            None,
        )
        self.assertEqual(row["status"], "oom")
        self.assertEqual(row["expected_outcome"], "ok")
        self.assertEqual(row["expectation_met"], "0")
        self.assertEqual(row["populated_context_proven"], "0")
        self.assertEqual(row["prefill_retention_vs_global_origin"], "")

    def test_global_origin_is_immutable_first_success(self) -> None:
        dma = {
            "scheme": q8.DMA,
            "profile": q8.SCREEN_PROFILE,
            "context_tokens": "4096",
            "status": "ok",
            "control_key": "pp:6",
            "case_time_utc": "2026-01-01T00:00:00Z",
            "is_global_origin": "1",
        }
        direct = {
            "scheme": q8.DIRECT,
            "profile": q8.SCREEN_PROFILE,
            "context_tokens": "4096",
            "status": "ok",
            "control_key": "pp:0",
            "case_time_utc": "2026-01-01T00:01:00Z",
            "is_global_origin": "",
        }
        self.assertIs(q8.global_origin([dma], q8.SCREEN_PROFILE, 4096), dma)
        self.assertIs(q8.global_origin([dma, direct], q8.SCREEN_PROFILE, 4096), dma)

    def test_frontier_probe_runs_shallow_then_actual_context_until_oom(self) -> None:
        runner = q8.Q8SweepRunner(
            self.manifest,
            "hash",
            Path("unused.csv"),
            Path("unused-frontier.csv"),
            self.model,
            dry_run=False,
            retry_errors=False,
            show_telemetry=False,
            max_cases=None,
        )
        visited = []

        def fake_run(case: q8.Q8Case):
            visited.append(case.context_tokens)
            status = "ok" if case.context_tokens <= 12288 else "oom"
            row = {
                "scheme": case.scheme,
                "phase": case.phase,
                "profile": case.profile,
                "control_key": case.control_key(),
                "context_tokens": str(case.context_tokens),
                "status": status,
                "prefill_retention_vs_global_origin": "1",
                "decode_retention_vs_global_origin": "1",
            }
            runner.rows[case.key()] = row
            return row

        runner.run_case = fake_run  # type: ignore[method-assign]
        frontier = runner.probe_capacity(
            q8.DMA,
            q8.PHASE_STAGING_COARSE,
            profile=q8.SCREEN_PROFILE,
            step_tokens=4096,
            seed_context=4096,
            target_pp=Decimal("6"),
        )
        self.assertEqual(frontier["context_tokens"], "12288")
        self.assertEqual(visited, [256, 4096, 8192, 12288, 16384])

    def test_page_search_bisects_to_exact_minimum_and_accepts_cuda_abort(self) -> None:
        runner = q8.Q8SweepRunner(
            self.manifest,
            "hash",
            Path("unused.csv"),
            Path("unused-frontier.csv"),
            self.model,
            dry_run=False,
            retry_errors=False,
            show_telemetry=False,
            max_cases=None,
        )
        visited = []

        def fake_run(case: q8.Q8Case):
            pages = int(case.page_budget or 0)
            visited.append(pages)
            row = {
                "scheme": case.scheme,
                "profile": case.profile,
                "context_tokens": str(case.context_tokens),
                "control_key": case.control_key(),
                "page_budget": str(pages),
                "status": "ok" if pages >= 37 else "cuda_capacity_fail",
            }
            runner.rows[case.key()] = row
            return row

        runner.run_case = fake_run  # type: ignore[method-assign]
        row = runner.find_min_vmm_pages(
            q8.DMA,
            q8.PHASE_STAGING_COARSE,
            q8.SCREEN_PROFILE,
            4096,
            seed_pages=30,
        )
        self.assertEqual(row["page_budget"], "37")
        self.assertIn(36, visited)

        case = q8.build_case(
            self.manifest, self.model, q8.DIRECT, q8.PHASE_DIRECT,
            q8.SCREEN_PROFILE, "frontier", 512,
        )
        aborted = base.CommandResult(134, "", "CUDA MMQ failure; Aborted", 1.0)
        status, _, _ = q8.classify_result(self.manifest, case, aborted, self.model)
        self.assertEqual(status, "cuda_capacity_fail")

    def test_runner_order(self) -> None:
        runner = q8.Q8SweepRunner(
            self.manifest,
            "hash",
            Path("unused.csv"),
            Path("unused-frontier.csv"),
            self.model,
            dry_run=True,
            retry_errors=False,
            show_telemetry=False,
            max_cases=None,
        )
        calls = []
        runner.run_direct = lambda: calls.append("direct")  # type: ignore[method-assign]
        runner.run_vmm_context_curve = (  # type: ignore[method-assign]
            lambda scheme, phase, **kwargs: calls.append((phase, scheme))
        )
        runner.run_native_context_curve = lambda: calls.append("native")  # type: ignore[method-assign]
        runner.coarse_context_stop = lambda scheme: 4096  # type: ignore[method-assign]
        runner.run()
        self.assertEqual(
            calls,
            [
                "direct",
                (q8.PHASE_STAGING_COARSE, q8.DMA),
                (q8.PHASE_ALTERNATES, q8.ZERO_COPY),
                "native",
                (q8.PHASE_FINE, q8.DMA),
                (q8.PHASE_FINE, q8.ZERO_COPY),
            ],
        )

    def test_frontier_csv_merges_fine_bracket_and_confirmation(self) -> None:
        common = {
            "schema_version": "1",
            "scheme": q8.DMA,
            "control_key": "pages:900",
            "target_offload_pp": "6.0",
            "native_ngl": "",
            "status": "ok",
            "prefill_tok_s": "2000",
            "decode_tok_s": "100",
            "page_budget": "900",
        }
        rows = [
            {**common, "phase": q8.PHASE_STAGING_COARSE, "profile": q8.SCREEN_PROFILE, "context_tokens": "8192"},
            {**common, "phase": q8.PHASE_FINE, "profile": q8.SCREEN_PROFILE, "context_tokens": "8448"},
            {**common, "phase": q8.PHASE_FINE, "profile": q8.SCREEN_PROFILE, "context_tokens": "8704", "status": "oom"},
            {
                **common,
                "phase": q8.PHASE_CONFIRM,
                "profile": q8.CONFIRM_PROFILE,
                "context_tokens": "8448",
                "prefill_tok_s": "1900",
                "decode_tok_s": "95",
                "prefill_retention_vs_global_origin": "0.09",
                "decode_retention_vs_global_origin": "0.08",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frontier.csv"
            q8.write_frontier_csv(path, rows)
            with path.open(newline="") as stream:
                saved = list(csv.DictReader(stream))
        selected = next(row for row in saved if row["frontier_context_tokens"] == "8448")
        self.assertEqual(selected["minimum_page_budget"], "900")
        self.assertEqual(selected["first_failed_context_tokens"], "")
        self.assertEqual(selected["confirmed_decode_tok_s"], "95")

    def test_resume_rejects_changed_model_or_manifest(self) -> None:
        row = q8.blank_row()
        row.update(
            manifest_sha256="wanted",
            model_file_bytes=str(self.model.file_bytes),
            eligible_tensor_bytes=str(self.model.eligible_tensor_bytes),
            gguf_tensor_count=str(self.model.tensor_count),
            model_context_length=str(self.model.context_length),
            model_metadata_sha256=self.model.metadata_sha256,
            model_sha256=self.model.file_sha256,
        )
        q8.validate_resume_rows([row], "wanted", self.model)
        with self.assertRaisesRegex(ValueError, "different manifest"):
            q8.validate_resume_rows([row], "other", self.model)

    def test_profile_backend_off_requires_deep(self) -> None:
        with mock.patch.object(q8, "load_manifest", return_value=(self.manifest, "hash")):
            with self.assertRaisesRegex(ValueError, "requires --profile deep"):
                q8.main(
                    [
                        "--manifest", "manifest.json",
                        "--profile", "light",
                        "--profile-backend-off",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
