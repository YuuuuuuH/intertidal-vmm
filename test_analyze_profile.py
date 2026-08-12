import csv
import json
import tempfile
import unittest
from pathlib import Path

import analyze_profile as profile


def fixture_record() -> dict:
    return {
        "schema_version": 1,
        "protocol": "intertidal-profile-v1",
        "run_id": "fixture-run",
        "status": "ok",
        "profile_mode": "deep",
        "elapsed_s": 0.010,
        "case_key": {
            "scheme": "intertidal_dma",
            "phase": "confirm",
            "control": "pages:64",
            "context_tokens": 4096,
            "page_budget": 64,
        },
        "backend": {
            "dropped": 0,
            "dma": {
                "count": 4,
                "bytes": 100_000_000,
                "sampled_count": 4,
                "sampled_bytes": 100_000_000,
                "copy_total_ms": 2.0,
                "timing_is_sampled": False,
                "effective_gbps_decimal": 50.0,
                "copy_p50_ms": 0.45,
                "copy_p95_ms": 0.60,
                "kick_to_ready_p95_ms": 0.75,
                "wait_count": 4,
                "wait_total_ms": 2.0,
                "wait_p95_ms": 0.70,
                "wait_max_ms": 0.80,
                "overlap_pct": 75.0,
                "prefetch_hit_pct": 100.0,
            },
            "zero_copy": {
                "touches": 0,
                "bytes": 0,
                "sampled_touches": 0,
                "sampled_bytes": 0,
            },
            "evals": [
                {"eval": 1, "work_ms": 3.0, "wait_ms": 0.5, "queue_ms": 0.1},
                {"eval": 2, "work_ms": 5.0, "wait_ms": 1.5, "queue_ms": 0.3},
            ],
            "layers": [
                {"layer": 7, "work_ms": 3.0, "wait_ms": 0.5, "queue_ms": 0.1},
                {"layer": 12, "work_ms": 5.0, "wait_ms": 1.5, "queue_ms": 0.3},
            ],
            "slowest_layers_by_wait_ms": [
                {"layer": 12, "wait_ms": 1.5},
                {"layer": 7, "wait_ms": 0.5},
            ],
        },
        "telemetry": {
            "gpu": {
                "compute_util_pct": {"mean": 91.0},
                "memory_util_pct": {"mean": 80.0},
                "graphics_clock_mhz": {"min": 2000.0},
            },
            "pcie": {
                "rx_mib_s": {"mean": 40_000.0, "max": 50_000.0},
                "tx_mib_s": {"max": 120.0},
            },
            "process": {"cpu_pct": {"max": 25.0}, "major_faults_delta": 0},
            "host": {
                "psi_avg10_pct": {
                    "memory_some": {"max": 0.0},
                    "memory_full": {"max": 0.0},
                }
            },
        },
    }


class AnalyzeProfileTests(unittest.TestCase):
    def test_fixed_ceiling_computes_utilization_wait_and_slowest_layer(self) -> None:
        row = profile.summarize_record(
            fixture_record(),
            source=Path("fixture.jsonl"),
            line_number=1,
            calibration=profile.H2DCalibration.fixed(62.5),
        )
        self.assertEqual(row["dma_pcie_utilization_pct"], 80.0)
        self.assertEqual(row["h2d_calibration_method"], "aggregate_average_copy_size")
        self.assertEqual(row["case_wall_remote_gbps_decimal"], 10.0)
        self.assertEqual(row["backend_work_ms"], 8.0)
        self.assertEqual(row["backend_wait_ms"], 2.0)
        self.assertEqual(row["backend_window_ms"], 10.0)
        self.assertEqual(row["backend_window_remote_gbps_decimal"], 10.0)
        self.assertEqual(row["exposed_wait_pct"], 20.0)
        self.assertEqual(row["wait_per_dma_us"], 500.0)
        self.assertEqual(row["slowest_layer"], 12)
        self.assertEqual(row["slowest_eval"], 2)
        self.assertAlmostEqual(row["telemetry_pcie_rx_peak_gbps_decimal"], 52.4288)
        self.assertAlmostEqual(row["telemetry_rx_peak_utilization_pct"], 83.88608)

    def test_mixed_ceiling_is_layer_sampled_byte_weighted_harmonic(self) -> None:
        record = fixture_record()
        record["backend"]["dma"].update(
            {
                "count": 2,
                "bytes": 30_000_000,
                "sampled_count": 2,
                "sampled_bytes": 30_000_000,
                "effective_gbps_decimal": 45.0,
                "timing_is_sampled": False,
            }
        )
        record["backend"]["layers"] = [
            {
                "layer": 1,
                "copy_count": 100,
                "total_bytes": 900_000_000,
                "sampled_copy_count": 1,
                "sampled_bytes": 10_000_000,
                "wait_ms": 0.5,
            },
            {
                "layer": 2,
                "copy_count": 100,
                "total_bytes": 900_000_000,
                "sampled_copy_count": 1,
                "sampled_bytes": 20_000_000,
                "wait_ms": 1.5,
            },
        ]
        calibration = profile.H2DCalibration(
            [
                profile.CalibrationPoint(40.0, 10_000_000, "10M single"),
                profile.CalibrationPoint(60.0, 20_000_000, "20M single"),
                profile.CalibrationPoint(50.0, 10_000_000, "10M train", "sustained"),
                profile.CalibrationPoint(70.0, 20_000_000, "20M train", "sustained"),
            ]
        )
        row = profile.summarize_record(
            record,
            source=Path("fixture.jsonl"),
            line_number=1,
            calibration=calibration,
        )
        expected_single = 30_000_000 / (10_000_000 / 40.0 + 20_000_000 / 60.0)
        expected_sustained = 30_000_000 / (
            10_000_000 / 50.0 + 20_000_000 / 70.0
        )
        self.assertEqual(row["h2d_calibration_method"], "layer_sampled_bytes_harmonic")
        self.assertAlmostEqual(row["h2d_ceiling_gbps_decimal"], expected_single)
        self.assertAlmostEqual(
            row["dma_pcie_utilization_pct"], 100.0 * 45.0 / expected_single
        )
        self.assertAlmostEqual(
            row["h2d_sustained_ceiling_gbps_decimal"], expected_sustained
        )
        self.assertAlmostEqual(
            row["dma_sustained_utilization_pct"], 100.0 * 45.0 / expected_sustained
        )

    def test_calibration_jsonl_interpolates_by_log_transfer_size(self) -> None:
        records = [
            {
                "protocol": "intertidal-pcie-calibration-v1",
                "direction": "h2d",
                "transfer_bytes": 8 * 1024 * 1024,
                "median_gbps_decimal": 40.0,
            },
            {
                "protocol": "intertidal-pcie-calibration-v1",
                "direction": "host_to_device",
                "transfer_bytes": 32 * 1024 * 1024,
                "p50_gbps": 60.0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in records) + "\n")
            calibration = profile.load_calibration(path)
        small = calibration.choose(7 * 1024 * 1024)
        large = calibration.choose(25 * 1024 * 1024)
        self.assertEqual(small.transfer_bytes, 8 * 1024 * 1024)
        self.assertEqual(small.gbps, 40.0)
        self.assertEqual(large.transfer_bytes, 25 * 1024 * 1024)
        expected = 40.0 + (60.0 - 40.0) * (
            profile.math.log2(25 / 8) / profile.math.log2(32 / 8)
        )
        self.assertAlmostEqual(large.gbps, expected)
        self.assertIn("log-size interpolation", large.source)

    def test_calibration_accepts_cuda_calibrator_result_schema(self) -> None:
        record = {
            "protocol": "intertidal-pcie-calibration-v1",
            "event": "result",
            "direction": "h2d",
            "bytes": 25_000_000,
            "single_p50_gbps_decimal": 51.25,
            "sustained_gbps_decimal": 58.75,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.jsonl"
            path.write_text(json.dumps(record) + "\n")
            choice = profile.load_calibration(path).choose(25_000_000)
        # The backend also times one DMA operation at a time, so single-copy
        # p50 is the like-for-like denominator rather than the copy-train rate.
        self.assertEqual(choice.gbps, 51.25)
        self.assertEqual(choice.transfer_bytes, 25_000_000)
        self.assertEqual(choice.sustained_gbps, 58.75)

    def test_nested_calibration_summary_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.jsonl"
            path.write_text(json.dumps({"summary": {"h2d_ceiling_gbps": 55.5}}) + "\n")
            choice = profile.load_calibration(path).choose(None)
        self.assertEqual(choice.gbps, 55.5)
        self.assertIsNone(choice.transfer_bytes)

    def test_explicit_ceiling_is_fallback_outside_calibrated_size_range(self) -> None:
        calibration = profile.H2DCalibration(
            [profile.CalibrationPoint(40.0, 8 * 1024 * 1024, "fixture")],
            fallback_gbps=62.5,
        )
        choice = calibration.choose(64 * 1024 * 1024)
        self.assertEqual(choice.gbps, 62.5)
        self.assertEqual(choice.source, "command line fallback")

    def test_cli_writes_csv_markdown_and_dependency_free_svg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar = root / "profile.jsonl"
            sidecar.write_text(json.dumps(fixture_record()) + "\n")
            prefix = root / "report"
            returncode = profile.main(
                [
                    str(sidecar),
                    "--h2d-ceiling-gbps",
                    "62.5",
                    "--output-prefix",
                    str(prefix),
                    "--title",
                    "Fixture report",
                ]
            )
            self.assertEqual(returncode, 0)
            csv_path, markdown_path, svg_path = profile.output_paths(prefix)
            self.assertTrue(csv_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertTrue(svg_path.exists())
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["dma_pcie_utilization_pct"]), 80.0)
            markdown = markdown_path.read_text()
            self.assertIn("DMA PCIe 利用率", markdown)
            self.assertIn("L12 / 1.500 ms", markdown)
            svg = svg_path.read_text()
            self.assertIn("<svg", svg)
            self.assertIn("PCIe utilization", svg)

    def test_zero_copy_does_not_claim_zero_dma_utilization(self) -> None:
        record = fixture_record()
        record["backend"]["dma"].update(
            {
                "count": 0,
                "bytes": 0,
                "sampled_count": 0,
                "sampled_bytes": 0,
                "effective_gbps_decimal": None,
            }
        )
        record["backend"]["zero_copy"] = {
            "touches": 4,
            "bytes": 100_000_000,
            "sampled_touches": 4,
            "sampled_bytes": 100_000_000,
        }
        row = profile.summarize_record(
            record,
            source=Path("fixture.jsonl"),
            line_number=1,
            calibration=profile.H2DCalibration.fixed(62.5),
        )
        self.assertIsNone(row["dma_pcie_utilization_pct"])
        self.assertEqual(row["backend_window_remote_gbps_decimal"], 10.0)
        self.assertIn("zero-copy", row["diagnosis"])

    def test_wrong_protocol_has_actionable_error(self) -> None:
        record = fixture_record()
        record["protocol"] = "something-else"
        with self.assertRaisesRegex(ValueError, "expected protocol"):
            profile.summarize_record(
                record,
                source=Path("fixture.jsonl"),
                line_number=3,
                calibration=profile.H2DCalibration.fixed(62.5),
            )


if __name__ == "__main__":
    unittest.main()
