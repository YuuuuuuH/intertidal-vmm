from __future__ import annotations

import csv
import contextlib
import io
import json
import tempfile
import unittest
from unittest import mock
from decimal import Decimal
from pathlib import Path

import gemma_capacity_sweep as sweep


SAMPLE_HEADER = (
    "build_commit,build_number,cuda,opencl,vulkan,kompute,metal,sycl,gpu_blas,"
    "blas,cpu_info,gpu_info,model_filename,model_type,model_size,model_n_params,"
    "n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,"
    "split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,"
    "tensor_buft_overrides,use_mmap,use_direct_io,embeddings,no_op_offload,no_host,"
    "fit_target,fit_min_ctx,n_ctx,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,"
    "avg_ts,stddev_ts\n"
)


def bench_record(n_prompt: int, n_gen: int, avg_ts: float, stddev_ts: float) -> str:
    values = [
        "abc", "1", "1", "0", "0", "0", "0", "1", "0", "cpu", "gpu",
        "model.gguf", "31B", "17322412272", "31000000000", "512", "512", "8",
        "", "0", "50", "f16", "f16", "99", "0", "0", "0", "1", "0", "",
        "", "1", "0", "0", "0", "0", "0", "0", "0", "167936", str(n_prompt),
        str(n_gen), "0", "2026-08-12T00:00:00Z", "1", "0", str(avg_ts), str(stddev_ts),
    ]
    return next(iter([",".join(json.dumps(value) for value in values)])) + "\n"


def checksum_log(
    repetitions: int = 5,
    *,
    warmup: bool = True,
    mismatch_index: int | None = None,
    nonfinite_index: int | None = None,
    include_nonfinite: bool = True,
) -> str:
    count = repetitions + int(warmup)
    entries = [("prompt", index + 1) for index in range(count)] + [
        ("decode", index + 100) for index in range(count)
    ]
    lines = []
    for index, (label, value) in enumerate(entries):
        checksum = value + (999 if index == mismatch_index else 0)
        suffix = ""
        if include_nonfinite:
            suffix = f",nonfinite={1 if index == nonfinite_index else 0}"
        lines.append(f"logit_checksum,{label},{checksum:016x}{suffix}")
    return "\n".join(lines) + "\n"


class CapacitySweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).with_name("capacity-sweep-manifest.example.json")
        cls.manifest = json.loads(path.read_text())

    def test_context_map_and_calibrated_grid(self) -> None:
        self.assertEqual(sweep.context_for_workset(self.manifest, Decimal("31")), 167936)
        self.assertEqual(sweep.context_for_workset(self.manifest, Decimal("32")), 180992)
        self.assertEqual(sweep.context_for_workset(self.manifest, Decimal("33")), 193792)
        self.assertEqual(sweep.context_for_workset(self.manifest, Decimal("34")), 206848)
        self.assertEqual(sweep.context_for_workset(self.manifest, Decimal("31.3")), 171776)
        self.assertEqual(len(sweep.workset_values(self.manifest)), 31)

    def test_page_budget_uses_binary_page_and_half_up_rounding(self) -> None:
        self.assertEqual(sweep.page_budget_for_pp(self.manifest, Decimal("0")), 0)
        self.assertEqual(sweep.page_budget_for_pp(self.manifest, Decimal("0.1")), 8)
        self.assertEqual(sweep.page_budget_for_pp(self.manifest, Decimal("1.0")), 83)

    def test_frontier_starts_at_gross_lower_bound(self) -> None:
        self.assertEqual(sweep.frontier_start_pp(self.manifest, Decimal("31")), Decimal("0.0"))
        self.assertEqual(sweep.frontier_start_pp(self.manifest, Decimal("32")), Decimal("4.5"))
        self.assertEqual(sweep.frontier_start_pp(self.manifest, Decimal("33")), Decimal("10.6"))
        self.assertEqual(sweep.frontier_start_pp(self.manifest, Decimal("34")), Decimal("16.8"))

    def test_commands_select_three_distinct_paths(self) -> None:
        workset = Decimal("31")
        dma = sweep.build_case(
            self.manifest, "intertidal_dma", "dense", workset, target_pp=Decimal("3.2")
        )
        zero = sweep.build_case(
            self.manifest, "cuda_zero_copy", "dense", workset, target_pp=Decimal("3.2")
        )
        native = sweep.build_case(
            self.manifest, "native_cpu_layers", "native", workset, native_ngl=58
        )
        dma_env, dma_command = sweep.command_for_case(self.manifest, dma)
        zero_env, zero_command = sweep.command_for_case(self.manifest, zero)
        native_env, native_command = sweep.command_for_case(self.manifest, native)
        self.assertEqual(dma_env["GGML_CUDA_HYBRID_MODE"], "staging")
        self.assertEqual(zero_env["GGML_CUDA_HYBRID_MODE"], "zero_copy")
        self.assertIn("GGML_CUDA_HYBRID_PAGE_BUDGET", dma_env)
        self.assertIn("-ot", dma_command)
        self.assertIn("-ot", zero_command)
        self.assertEqual(native_env, {"LLAMA_BENCH_LOGIT_CHECKSUM": "1"})
        self.assertEqual(dma_env["LLAMA_BENCH_LOGIT_CHECKSUM"], "1")
        self.assertEqual(native_command[-2:], ["-ngl", "58"])
        self.assertIn("--ctx-size", dma_command)

    def test_zero_percent_is_true_all_local_control(self) -> None:
        case = sweep.build_case(
            self.manifest, "intertidal_dma", "coarse", Decimal("31"), target_pp=Decimal(0)
        )
        env, command = sweep.command_for_case(self.manifest, case)
        self.assertNotIn("GGML_CUDA_HYBRID_PAGE_BUDGET", env)
        self.assertEqual(env["LLAMA_BENCH_LOGIT_CHECKSUM"], "1")
        self.assertNotIn("-ot", command)

    def test_checksum_sequence_shape_is_all_warmups_then_repetitions(self) -> None:
        entries = sweep.parse_logit_checksums(checksum_log())
        self.assertEqual(len(entries), 12)
        self.assertEqual([entry.label for entry in entries], ["prompt"] * 6 + ["decode"] * 6)
        sweep.validate_checksum_shape(
            self.manifest,
            [entry.identity() for entry in entries],
            source="test",
        )

        no_warmup = json.loads(json.dumps(self.manifest))
        no_warmup["benchmark"]["no_warmup"] = True
        entries = sweep.parse_logit_checksums(checksum_log(warmup=False))
        self.assertEqual(len(entries), 10)
        self.assertEqual([entry.label for entry in entries], ["prompt"] * 5 + ["decode"] * 5)
        sweep.validate_checksum_shape(
            no_warmup,
            [entry.identity() for entry in entries],
            source="test",
        )

    def test_checksum_baseline_then_exact_match(self) -> None:
        case = sweep.build_case(
            self.manifest, "intertidal_dma", "coarse", Decimal("31"), target_pp=Decimal(0)
        )
        result = sweep.CommandResult(0, "", checksum_log(), 1.0)
        captured = sweep.evaluate_correctness(self.manifest, case, result, "ok", None)
        self.assertEqual(captured.status, "baseline_captured")
        self.assertEqual(captured.nonfinite_logits, 0)
        self.assertEqual(len(captured.baseline or []), 12)

        remote_case = sweep.build_case(
            self.manifest, "intertidal_dma", "dense", Decimal("31"), target_pp=Decimal("1")
        )
        matched = sweep.evaluate_correctness(
            self.manifest, remote_case, result, "ok", captured.baseline
        )
        self.assertEqual(matched.status, "match")

    def test_checksum_mismatch_nan_missing_and_legacy_format_fail(self) -> None:
        case = sweep.build_case(
            self.manifest, "intertidal_dma", "dense", Decimal("31"), target_pp=Decimal("1")
        )
        baseline_entries = sweep.parse_logit_checksums(checksum_log())
        baseline = [entry.identity() for entry in baseline_entries]

        scenarios = (
            (checksum_log(mismatch_index=7), "checksum mismatch"),
            (checksum_log(nonfinite_index=3), "non-finite logits"),
            ("\n".join(checksum_log().splitlines()[:-1]) + "\n", "sequence shape mismatch"),
            (checksum_log(include_nonfinite=False), "do not report nonfinite"),
        )
        for log, expected_error in scenarios:
            with self.subTest(expected_error=expected_error):
                result = sweep.CommandResult(0, "", log, 1.0)
                evaluated = sweep.evaluate_correctness(
                    self.manifest, case, result, "ok", baseline
                )
                self.assertEqual(evaluated.status, "fail")
                self.assertIn(expected_error, evaluated.error)

    def test_oom_does_not_require_checksum(self) -> None:
        case = sweep.build_case(
            self.manifest, "intertidal_dma", "dense", Decimal("33"), target_pp=Decimal("10")
        )
        result = sweep.CommandResult(1, "", "CUDA out of memory", 1.0)
        evaluated = sweep.evaluate_correctness(self.manifest, case, result, "oom", None)
        self.assertEqual(evaluated.status, "not_run")

    def test_correctness_failure_is_persisted_then_aborts_runner(self) -> None:
        baseline_entries = sweep.parse_logit_checksums(checksum_log())
        baseline = [entry.identity() for entry in baseline_entries]
        failed_result = sweep.CommandResult(
            returncode=0,
            stdout=SAMPLE_HEADER
            + bench_record(512, 0, 2100.0, 10.0)
            + bench_record(0, 64, 70.0, 0.3),
            stderr=checksum_log(mismatch_index=4),
            elapsed_s=1.0,
        )
        case = sweep.build_case(
            self.manifest,
            "intertidal_dma",
            "dense",
            Decimal("31"),
            target_pp=Decimal("1"),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            runner = sweep.SweepRunner(
                self.manifest,
                "hash",
                output,
                dry_run=False,
                retry_errors=False,
                show_telemetry=False,
                max_cases=None,
            )
            runner.checksum_baseline = baseline
            with runner, mock.patch.object(
                sweep, "execute_wrapper", return_value=failed_result
            ), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "correctness failure"):
                    runner.run_case(case)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "correctness_fail")
            self.assertEqual(rows[0]["checksum_status"], "fail")
            self.assertIn("checksum mismatch", rows[0]["error_tail"])

    def test_initial_control_self_normalizes(self) -> None:
        case = sweep.build_case(
            self.manifest,
            "intertidal_dma",
            "coarse",
            Decimal("31"),
            target_pp=Decimal(0),
        )
        result = sweep.CommandResult(
            returncode=0,
            stdout=SAMPLE_HEADER
            + bench_record(512, 0, 2200.0, 10.0)
            + bench_record(0, 64, 74.0, 0.3),
            stderr="",
            elapsed_s=1.0,
        )
        row = sweep.row_for_result(
            self.manifest,
            "hash",
            "run",
            case,
            {},
            ["bench"],
            result,
            None,
        )
        self.assertEqual(row["prefill_retention"], "1.000000")
        self.assertEqual(row["decode_retention"], "1.000000")
        self.assertEqual(row["baseline_source_workset_gib"], "31")

    def test_parse_bench(self) -> None:
        output = SAMPLE_HEADER + bench_record(512, 0, 2190.11, 12.5) + bench_record(0, 64, 73.0, 0.4)
        parsed = sweep.parse_bench(output)
        self.assertAlmostEqual(parsed["prefill_tok_s"], 2190.11)
        self.assertAlmostEqual(parsed["decode_tok_s"], 73.0)

    def test_parse_layout_and_native_buffers(self) -> None:
        stderr = (
            "hybrid_vmm,mode=staging,requested_pages=266,gross_pages=266,"
            "gross_bytes=557842432,staging_bytes=18874368,net_bytes=538006323,"
            "selected_tensors=260,layers=60\n"
            "load: CPU_Mapped model buffer size = 2075.48 MiB\n"
            "load: CUDA0 model buffer size = 15200.53 MiB\n"
        )
        layout = sweep.parse_hybrid_layout(stderr)
        native = sweep.parse_native_buffers(stderr)
        self.assertEqual(layout["gross_pages"], 266)
        self.assertEqual(layout["selected_layers"], 60)
        self.assertAlmostEqual(native["native_cpu_model_mib"], 2075.48)
        self.assertAlmostEqual(native["native_cuda_model_mib"], 15200.53)

    def test_native_capacity_uses_same_context_measured_peak_delta(self) -> None:
        case = sweep.build_case(
            self.manifest,
            "native_cpu_layers",
            "capacity_frontier",
            Decimal("31"),
            native_ngl=58,
        )
        result = sweep.CommandResult(
            returncode=0,
            stdout=SAMPLE_HEADER
            + bench_record(512, 0, 1200.0, 10.0)
            + bench_record(0, 64, 40.0, 0.3),
            stderr="load: CPU_Mapped model buffer size = 900.00 MiB\n",
            elapsed_s=1.0,
            telemetry=[
                sweep.TelemetrySample(1000, 42, 30100.0, 30200.0, None, None, None, None)
            ],
        )
        row = sweep.row_for_result(
            self.manifest,
            "hash",
            "run",
            case,
            {},
            ["bench"],
            result,
            None,
            native_reference_peak_mib=31700.0,
        )
        self.assertEqual(row["net_saved_mib"], "1600.000")
        self.assertEqual(row["logical_working_set_mib"], "31700.000")
        self.assertEqual(row["all_local_reference_peak_mib"], "31700.000")
        self.assertEqual(
            row["net_saved_source"], "measured_same_context_all_local_peak_delta"
        )
        self.assertEqual(row["gross_remote_mib"], "")

    def test_native_overphysical_capacity_is_explicitly_inferred(self) -> None:
        case = sweep.build_case(
            self.manifest,
            "native_cpu_layers",
            "capacity_frontier",
            Decimal("33"),
            native_ngl=55,
        )
        result = sweep.CommandResult(
            returncode=0,
            stdout=SAMPLE_HEADER
            + bench_record(512, 0, 900.0, 10.0)
            + bench_record(0, 64, 30.0, 0.3),
            stderr="",
            elapsed_s=1.0,
            telemetry=[
                sweep.TelemetrySample(1000, 42, 31000.0, 31100.0, None, None, None, None)
            ],
        )
        row = sweep.row_for_result(
            self.manifest,
            "hash",
            "run",
            case,
            {},
            ["bench"],
            result,
            None,
        )
        calibrated = sweep.calibrated_peak_mib(self.manifest, case.context_tokens)
        self.assertEqual(
            row["net_saved_source"], "inferred_calibrated_all_local_peak_delta"
        )
        self.assertAlmostEqual(float(row["logical_working_set_mib"]), calibrated, places=3)
        self.assertAlmostEqual(
            float(row["net_saved_mib"]), calibrated - 31000.0, places=3
        )
        self.assertEqual(row["gross_remote_mib"], "")

    def test_native_reference_lookup_requires_same_context(self) -> None:
        current_context = sweep.context_for_workset(self.manifest, Decimal("31"))
        rows = {
            ("native_cpu_layers", "31", "ngl:61"): {
                "scheme": "native_cpu_layers",
                "status": "ok",
                "native_ngl": "61",
                "context_tokens": str(current_context),
                "gpu_process_peak_mib": "31700.0",
            },
            ("native_cpu_layers", "32", "ngl:61"): {
                "scheme": "native_cpu_layers",
                "status": "ok",
                "native_ngl": "61",
                "context_tokens": str(current_context + 256),
                "gpu_process_peak_mib": "32000.0",
            },
        }
        self.assertEqual(
            sweep.native_all_local_peak_from_rows(
                self.manifest, rows, current_context
            ),
            31700.0,
        )

    def test_telemetry_parser_and_summary(self) -> None:
        lines = [
            "__GEMMA_SWEEP_GPU__|1000|42|17566|18000|2400|63|420.5|98\n",
            "__GEMMA_SWEEP_GPU__|1100|42|17600|18100|2500|65|430.5|100\n",
        ]
        samples = [sweep.parse_telemetry_line(line) for line in lines]
        self.assertTrue(all(sample is not None for sample in samples))
        summary = sweep.summarize_telemetry([sample for sample in samples if sample])
        self.assertEqual(summary["gpu_process_peak_mib"], "17600.000")
        self.assertEqual(summary["graphics_clock_mean_mhz"], "2450.000")
        self.assertEqual(summary["temperature_max_c"], "65.000")
        self.assertEqual(summary["power_max_w"], "430.500")

    def test_profile_system_telemetry_and_backend_summary(self) -> None:
        system = sweep.parse_telemetry_line(
            "__GEMMA_SWEEP_SYSTEM__|epoch_ms=1100|pid=42|proc_rss_mib=101.5|"
            "proc_locked_mib=64|proc_major_faults=9|proc_cpu_ticks=120|"
            "total_cpu_ticks=2200|cpu_count=16|proc_read_mib=3|proc_write_mib=4|"
            "mem_available_mib=32768|swap_free_mib=1024|psi_cpu_some_pct=1.2|"
            "psi_memory_some_pct=0.2|psi_memory_full_pct=0|psi_io_some_pct=0.4|"
            "psi_io_full_pct=0.1|memory_util_pct=77|memory_clock_mhz=14000|"
            "pstate=P0|pcie_link_gen=5|pcie_link_width=16|pcie_rx_kib_s=2048|"
            "pcie_tx_kib_s=1024\n"
        )
        self.assertIsNotNone(system)
        assert system is not None
        self.assertEqual(system.pcie_rx_mib_s, 2.0)
        self.assertEqual(system.pstate, "P0")
        summary = sweep.telemetry_profile_summary([system])
        self.assertEqual(summary["process"]["rss_mib"]["max"], 101.5)
        self.assertEqual(summary["host"]["psi_avg10_pct"]["cpu_some"]["max"], 1.2)

        lines = [
            "hybrid_profile,event=config,mode=staging,sample_every=1",
            "hybrid_profile,event=dma,seq=1,bytes=1048576,copy_ms=0.1,kick_to_ready_ms=0.2",
            "hybrid_profile,event=dma,seq=2,bytes=1048576,copy_ms=0.3,kick_to_ready_ms=0.4",
            "hybrid_profile,event=wait,seq=2,wait_ms=0.05",
        ]
        events = [sweep.parse_backend_profile_line(line) for line in lines]
        backend = sweep.backend_profile_summary([event for event in events if event])
        self.assertEqual(backend["dma"]["count"], 2)
        self.assertAlmostEqual(backend["dma"]["effective_gbps_decimal"], 5.24288)
        self.assertAlmostEqual(backend["dma"]["copy_p95_ms"], 0.29)
        self.assertEqual(backend["dma"]["wait_count"], 1)

    def test_profile_config_defaults_off_and_deep_enables_backend(self) -> None:
        config = sweep.profiling_config(self.manifest)
        self.assertEqual(config["mode"], "off")
        env: dict[str, str] = {}
        sweep.enable_backend_profile(env, config)
        self.assertNotIn("GGML_CUDA_HYBRID_PROFILE", env)
        deep = sweep.profiling_config(self.manifest, "deep")
        sweep.enable_backend_profile(env, deep)
        self.assertEqual(env["GGML_CUDA_HYBRID_PROFILE"], "1")
        self.assertEqual(env["GGML_CUDA_HYBRID_FORCE_DIRECT"], "1")
        self.assertEqual(env["GGML_CUDA_HYBRID_PROFILE_RING"], "256")
        self.assertTrue(deep["force_direct"])
        self.assertTrue(deep["pcie_dmon"])
        self.assertGreaterEqual(deep["interval_s"], 0.2)
        light = sweep.profiling_config(self.manifest, "light")
        self.assertFalse(light["force_direct"])
        self.assertFalse(light["pcie_dmon"])
        self.assertGreaterEqual(light["interval_s"], 1.0)

        self.manifest["profiling"]["force_direct"] = True
        direct_control = sweep.profiling_config(self.manifest, "light")
        direct_env: dict[str, str] = {}
        sweep.enable_backend_profile(direct_env, direct_control)
        self.assertEqual(direct_env["GGML_CUDA_HYBRID_FORCE_DIRECT"], "1")
        self.assertNotIn("GGML_CUDA_HYBRID_PROFILE", direct_env)

    def test_selected_profile_cli_targets_prepend_checksum_control(self) -> None:
        with mock.patch.object(sweep, "load_manifest", return_value=(self.manifest, "a" * 64)), \
             mock.patch.object(sweep, "SweepRunner") as runner_type, \
             mock.patch.object(sweep, "plan_summary", return_value={}):
            runner = runner_type.return_value.__enter__.return_value
            runner.reached_limit.return_value = False
            result = sweep.main(
                [
                    "--manifest", "manifest.json",
                    "--output", "selected.csv",
                    "--schemes", "intertidal_dma",
                    "--only-offload-pp", "3.2,10",
                    "--profile", "deep",
                    "--profile-benchmark", "decode",
                ]
            )
        self.assertEqual(result, 0)
        targets = [call.args[0].target_offload_pp for call in runner.run_case.call_args_list]
        self.assertEqual(targets, [Decimal("0"), Decimal("3.2"), Decimal("10")])
        self.assertTrue(
            all(call.args[0].phase == "profile_probe_decode" for call in runner.run_case.call_args_list)
        )
        manifest = runner_type.call_args.args[0]
        self.assertEqual(manifest["benchmark"]["prompt"], 0)
        self.assertGreater(manifest["benchmark"]["generation"], 0)

    def test_profile_benchmark_requires_selected_cases(self) -> None:
        with mock.patch.object(sweep, "load_manifest", return_value=(self.manifest, "a" * 64)):
            with self.assertRaisesRegex(ValueError, "requires --only-offload-pp"):
                sweep.main(
                    [
                        "--manifest", "manifest.json",
                        "--output", "selected.csv",
                        "--profile-benchmark", "prefill",
                    ]
                )

    def test_profile_backend_off_requires_deep(self) -> None:
        with mock.patch.object(sweep, "load_manifest", return_value=(self.manifest, "a" * 64)):
            with self.assertRaisesRegex(ValueError, "requires --profile deep"):
                sweep.main(
                    [
                        "--manifest", "manifest.json",
                        "--output", "selected.csv",
                        "--profile", "light",
                        "--profile-backend-off",
                    ]
                )

    def test_profile_sidecar_is_one_append_only_record_per_executed_case(self) -> None:
        profile = sweep.profiling_config(self.manifest, "light")
        result = sweep.CommandResult(
            0,
            "",
            "",
            0.5,
            started_epoch_ms=1000,
            ended_epoch_ms=1500,
        )
        record = sweep.profile_sidecar_record(
            profile=profile,
            run_id="run",
            manifest_sha256="a" * 64,
            case_key={"scheme": "intertidal_dma", "control": "pp:1"},
            status="ok",
            result=result,
            command_identity={"env": {}, "argv": ["bench"]},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            sweep.append_profile_sidecar(path, record)
            sweep.append_profile_sidecar(path, record)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            loaded = [json.loads(line) for line in lines]
            self.assertEqual(loaded[0]["protocol"], "intertidal-profile-v1")
            self.assertEqual(loaded[0]["manifest_sha256"], "a" * 64)
            self.assertEqual(loaded[0]["command_sha256"], loaded[1]["command_sha256"])

    def test_backend_profile_real_v1_schema_uses_total_without_double_counting(self) -> None:
        # Literal field order/schema emitted by ggml-cuda-hostmapped.cu.  The
        # window and total records overlap; only scope=total is authoritative.
        lines = [
            "hybrid_profile,version=1,event=config,mode=staging,profile_scope=direct_streams,"
            "graphs=forced_off,device=0,sample_every=2,summary_every=128,"
            "force_direct_source=profile,detail=1,ring=128,remote_bytes=10485760,layers=2",
            "hybrid_profile,version=1,event=summary,scope=window,window=1,mode=staging,"
            "profile_scope=direct_streams,graphs=forced_off,sample_every=2,eval_count=1,"
            "graph_eval_count=0,prepare_count=3,prefetch_count=3,copy_count=3,wait_count=3,"
            "dma_count=3,dma_bytes=3000000,dma_ms=0.300000,prefetch_hits=2,prefetch_misses=0,"
            "zero_copy_touches=0,zero_copy_bytes=0,estimated_read_bytes=0,zc_touches=0,zc_bytes=0,"
            "total_bytes=3000000,sampled_copy_count=2,sampled_bytes=2000000,sampled_zero_copy=0,"
            "sampled_zero_copy_bytes=0,copy_ms=0.200000,ready_ms=0.250000,queue_ms=0.050000,"
            "wait_ms=0.010000,hidden_ms=0.240000,work_ms=1.000000,work_window_ms=1.000000,"
            "effective_gbps=10.000000,overlap_pct=96.000,hidden_pct=96.000,stall_pct=0.990,dropped=0",
            "hybrid_profile,version=1,event=summary,scope=total,window=1,mode=staging,"
            "profile_scope=direct_streams,graphs=forced_off,sample_every=2,eval_count=2,"
            "graph_eval_count=0,prepare_count=5,prefetch_count=5,copy_count=5,wait_count=5,"
            "dma_count=5,dma_bytes=5000000,dma_ms=0.400000,prefetch_hits=3,prefetch_misses=1,"
            "zero_copy_touches=0,zero_copy_bytes=0,estimated_read_bytes=0,zc_touches=0,zc_bytes=0,"
            "total_bytes=5000000,sampled_copy_count=4,sampled_bytes=4000000,sampled_zero_copy=0,"
            "sampled_zero_copy_bytes=0,copy_ms=0.400000,ready_ms=0.500000,queue_ms=0.100000,"
            "wait_ms=0.100000,hidden_ms=0.400000,work_ms=2.000000,work_window_ms=2.000000,"
            "effective_gbps=10.000000,overlap_pct=80.000,hidden_pct=80.000,stall_pct=4.762,dropped=1",
            "hybrid_profile,version=1,event=layer,mode=staging,profile_scope=direct_streams,layer=7,"
            "parity=1,prepare_count=3,prefetch_count=3,copy_count=3,dma_count=3,dma_bytes=3000000,"
            "dma_ms=0.300000,prefetch_hits=2,prefetch_misses=1,wait_count=3,zero_copy_touches=0,"
            "zero_copy_bytes=0,estimated_read_bytes=0,zc_touches=0,zc_bytes=0,total_bytes=3000000,"
            "sampled_copy_count=2,sampled_bytes=2000000,sampled_zero_copy=0,"
            "sampled_zero_copy_bytes=0,sampled_work_count=2,copy_ms=0.200000,ready_ms=0.250000,"
            "queue_ms=0.050000,wait_ms=0.080000,hidden_ms=0.170000,work_ms=1.000000,"
            "work_window_ms=1.000000,effective_gbps=10.000000,overlap_pct=68.000,"
            "hidden_pct=68.000,stall_pct=7.407",
            "hybrid_profile,version=1,event=eval,scope=eval,window=0,eval=2,mode=staging,"
            "profile_scope=direct_streams,graphs=forced_off,sample_every=2,eval_count=1,"
            "graph_eval_count=0,prepare_count=2,prefetch_count=2,copy_count=2,wait_count=2,"
            "dma_count=2,dma_bytes=2000000,dma_ms=0.200000,prefetch_hits=1,prefetch_misses=1,"
            "zero_copy_touches=0,zero_copy_bytes=0,estimated_read_bytes=0,zc_touches=0,zc_bytes=0,"
            "total_bytes=2000000,sampled_copy_count=1,sampled_bytes=1000000,sampled_zero_copy=0,"
            "sampled_zero_copy_bytes=0,copy_ms=0.100000,ready_ms=0.150000,queue_ms=0.020000,"
            "wait_ms=0.050000,hidden_ms=0.100000,work_ms=0.900000,work_window_ms=0.900000,"
            "effective_gbps=10.000000,overlap_pct=66.667,hidden_pct=66.667,stall_pct=5.263,dropped=0",
            "hybrid_profile,version=1,event=dma,mode=staging,profile_scope=direct_streams,"
            "seq=9,eval=1,work_eval=2,layer=7,parity=1,bytes=1000000,copy_ms=0.100000,"
            "kick_to_ready_ms=0.150000,ready_ms=0.150000,queue_ms=0.020000,hidden_ms=0.100000,"
            "effective_gbps=10.000000,hidden_pct=66.667",
        ]
        parsed = [sweep.parse_backend_profile_line(line) for line in lines]
        summary = sweep.backend_profile_summary([event for event in parsed if event])
        self.assertEqual(summary["aggregation_source"], "summary_total")
        self.assertEqual(summary["summary_records_seen"], 2)
        self.assertEqual(summary["summary_records_used"], 1)
        self.assertEqual(summary["dma"]["count"], 5)
        self.assertEqual(summary["dma"]["bytes"], 5_000_000)
        self.assertEqual(summary["dma"]["sampled_count"], 4)
        self.assertEqual(summary["dma"]["sampled_bytes"], 4_000_000)
        self.assertTrue(summary["dma"]["timing_is_sampled"])
        self.assertEqual(summary["dma"]["copy_total_ms"], 0.4)
        self.assertEqual(summary["dma"]["effective_gbps_decimal"], 10.0)
        self.assertEqual(summary["dma"]["hidden_pct"], 80.0)
        self.assertEqual(summary["dma"]["wait_total_ms"], 0.1)
        self.assertEqual(summary["dropped"], 1)
        self.assertEqual(summary["layers"][0]["layer"], 7)
        self.assertEqual(summary["layers"][0]["wait_ms"], 0.08)
        self.assertEqual(summary["evals"][0]["eval"], 2)
        self.assertEqual(summary["evals"][0]["wait_ms"], 0.05)
        self.assertEqual(summary["cross_eval_detail_samples"], 1)

    def test_backend_profile_legacy_sample_schema_remains_parseable(self) -> None:
        lines = [
            "hybrid_profile,version=1,event=sample,mode=staging,seq=1,layer=3,bytes=1000000,"
            "copy_ms=0.100000,ready_ms=0.200000,queue_ms=0.010000,wait_ms=0.020000,"
            "hidden_ms=0.180000,work_ms=0.500000,effective_gbps=10.000000,"
            "overlap_pct=90.000,stall_pct=3.846,prefetch_hit=1",
            "hybrid_profile,version=1,event=sample,mode=zero_copy,seq=2,layer=4,"
            "remote_bytes=2000000,work_ms=0.600000",
        ]
        parsed = [sweep.parse_backend_profile_line(line) for line in lines]
        summary = sweep.backend_profile_summary([event for event in parsed if event])
        self.assertEqual(summary["aggregation_source"], "detail_events")
        self.assertEqual(summary["dma"]["count"], 1)
        self.assertEqual(summary["dma"]["bytes"], 1_000_000)
        self.assertEqual(summary["dma"]["wait_count"], 1)
        self.assertEqual(summary["dma"]["prefetch_hits"], 1)
        self.assertEqual(summary["zero_copy"]["touches"], 1)
        self.assertEqual(summary["zero_copy"]["bytes"], 2_000_000)

    def test_oom_is_recordable_not_parse_error(self) -> None:
        result = sweep.CommandResult(
            returncode=1,
            stdout="",
            stderr="CUDA error: out of memory while allocating KV buffer",
            elapsed_s=1.0,
        )
        status, _, _ = sweep.classify_result(result)
        self.assertEqual(status, "oom")

    def test_ssh_auth_failure_is_transport_error(self) -> None:
        result = sweep.CommandResult(
            returncode=255,
            stdout="",
            stderr="wici@192.168.1.182: Permission denied (publickey,password).",
            elapsed_s=0.1,
        )
        status, _, _ = sweep.classify_result(result)
        self.assertEqual(status, "transport_error")

    def test_silent_ssh_exit_255_is_transport_error(self) -> None:
        result = sweep.CommandResult(
            returncode=255,
            stdout="",
            stderr="",
            elapsed_s=0.1,
        )
        status, parsed, error = sweep.classify_result(result)
        self.assertEqual(status, "transport_error")
        self.assertEqual(parsed, {})
        self.assertEqual(error, "SSH exited with status 255")

    def test_two_consecutive_slow_points_stop(self) -> None:
        rows = {}
        for target, retention in (("0", "1"), ("1", "0.11"), ("2", "0.10"), ("3", "0.09")):
            rows[("intertidal_dma", "31", f"pp:{target}")] = {
                "status": "ok",
                "prefill_retention": retention,
                "decode_retention": retention,
            }
        crossing = sweep.crossing_target(
            rows,
            "intertidal_dma",
            Decimal("31"),
            [Decimal(item) for item in ("0", "1", "2", "3")],
            0.10,
            2,
        )
        self.assertEqual(crossing, Decimal("3"))

    def test_phase_planner_has_one_full_curve_and_all_frontiers(self) -> None:
        runner = sweep.SweepRunner(
            self.manifest,
            "hash",
            Path("unused.csv"),
            dry_run=True,
            retry_errors=False,
            show_telemetry=False,
            max_cases=None,
        )
        calls = []
        runner.run_hybrid_curve = lambda scheme, ws: calls.append(("full-hybrid", scheme, ws))
        runner.run_native_curve = lambda ws: calls.append(("full-native", ws))
        runner.run_hybrid_frontier = lambda scheme, ws: calls.append(("frontier-hybrid", scheme, ws))
        runner.run_native_frontier = lambda ws: calls.append(("frontier-native", ws))
        runner.run()

        full_hybrid = [item for item in calls if item[0] == "full-hybrid"]
        full_native = [item for item in calls if item[0] == "full-native"]
        frontier_hybrid = [item for item in calls if item[0] == "frontier-hybrid"]
        frontier_native = [item for item in calls if item[0] == "frontier-native"]
        self.assertEqual(len(full_hybrid), 2)
        self.assertTrue(all(item[-1] == Decimal("31.0") for item in full_hybrid))
        self.assertEqual(full_native, [("full-native", Decimal("31.0"))])
        self.assertEqual(len(frontier_hybrid), 2 * 31)
        self.assertEqual(len(frontier_native), 31)

    def test_frontier_oom_then_fit_and_one_confirmation(self) -> None:
        runner = sweep.SweepRunner(
            self.manifest,
            "hash",
            Path("unused.csv"),
            dry_run=False,
            retry_errors=False,
            show_telemetry=False,
            max_cases=None,
        )
        visited = []
        required = sweep.frontier_required_mib(self.manifest, Decimal("32"))

        def fake_run(case):
            visited.append(case.target_offload_pp)
            if case.target_offload_pp in {Decimal("0"), Decimal("4.5")}:
                return {"status": "oom"}
            return {"status": "ok", "net_saved_mib": str(required + 1)}

        runner.run_case = fake_run
        runner.run_hybrid_frontier("intertidal_dma", Decimal("32"))
        self.assertEqual(
            visited,
            [Decimal("0"), Decimal("4.5"), Decimal("4.6"), Decimal("4.7")],
        )

    def test_native_frontier_rejects_first_ok_when_saving_is_insufficient(self) -> None:
        runner = sweep.SweepRunner(
            self.manifest,
            "hash",
            Path("unused.csv"),
            dry_run=False,
            retry_errors=False,
            show_telemetry=False,
            max_cases=None,
        )
        required = sweep.frontier_required_mib(self.manifest, Decimal("33"))
        visited = []

        def fake_run(case):
            visited.append(case.native_ngl)
            if case.native_ngl == 61:
                return {"status": "oom"}
            if case.native_ngl == 60:
                return {"status": "ok", "net_saved_mib": str(required - 1)}
            return {"status": "ok", "net_saved_mib": str(required + 1)}

        runner.run_case = fake_run
        runner.run_native_frontier(Decimal("33"))
        self.assertEqual(visited, [61, 60, 59, 58])

    def test_plan_summary_reports_adaptive_phases(self) -> None:
        summary = sweep.plan_summary(self.manifest)
        self.assertEqual(summary["pure_curve_workset_gib"], 31.0)
        self.assertEqual(summary["maximum_unique_points_per_full_hybrid_curve"], 951)
        self.assertEqual(len(summary["capacity_frontier"]), 31)
        self.assertEqual(summary["capacity_frontier"][0]["hybrid_start_pp"], 0.0)
        self.assertEqual(summary["capacity_frontier"][-1]["hybrid_start_pp"], 16.8)

    def test_cli_dry_run_summary_does_not_open_ssh(self) -> None:
        manifest_path = Path(__file__).with_name("capacity-sweep-manifest.example.json")
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            returncode = sweep.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--dry-run",
                    "--max-cases",
                    "1",
                    "--phases",
                    "capacity_frontier",
                ]
            )
        output = stream.getvalue()
        self.assertEqual(returncode, 0)
        self.assertIn('"capacity_frontier"', output)
        self.assertIn("remote_files_created", output)
        self.assertIn("case scheme=intertidal_dma", output)

    def test_resume_keeps_oom_as_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sweep.CSV_FIELDS)
                writer.writeheader()
                row = sweep.blank_row()
                row.update(
                    scheme="intertidal_dma",
                    workset_target_gib="33.0",
                    target_offload_pp="0.0",
                    status="oom",
                )
                writer.writerow(row)
            rows = sweep.read_resume_rows(path)
            loaded = rows[("intertidal_dma", "33", "pp:0")]
            self.assertTrue(sweep.is_completed(loaded, retry_errors=True))

    def test_remote_wrapper_has_no_result_file(self) -> None:
        wrapper = sweep.remote_wrapper(
            {"GGML_CUDA_HYBRID_MODE": "staging"},
            ["/tmp/bench", "-m", "/tmp/model with spaces.gguf"],
            0.1,
            "0",
        )
        self.assertIn("env -u LD_PRELOAD", wrapper)
        self.assertIn("ulimit -c 0", wrapper)
        self.assertIn("__GEMMA_SWEEP_GPU__", wrapper)
        self.assertNotIn("__GEMMA_SWEEP_SYSTEM__", wrapper)
        self.assertNotIn("mktemp", wrapper)
        self.assertNotIn(">/tmp", wrapper)

        profile = sweep.profiling_config(self.manifest, "deep")
        deep = sweep.remote_wrapper({}, ["/tmp/bench"], 0.2, "0", profile)
        self.assertIn("__GEMMA_SWEEP_SYSTEM__", deep)
        self.assertIn("pcie.rx_util,pcie.tx_util", deep)
        self.assertIn("/proc/pressure/memory", deep)


if __name__ == "__main__":
    unittest.main()
