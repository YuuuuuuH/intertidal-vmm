from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import plot_capacity_results as plot


FIELDS = [
    "run_id",
    "case_time_utc",
    "scheme",
    "phase",
    "workset_target_gib",
    "calibrated_all_local_mib",
    "target_offload_pp",
    "page_budget",
    "native_ngl",
    "status",
    "gross_remote_pp",
    "native_cpu_model_mib",
    "native_cuda_model_mib",
    "logical_working_set_mib",
    "prefill_retention",
    "decode_retention",
]


def row(**updates: str) -> dict[str, str]:
    result = {field: "" for field in FIELDS}
    result.update(
        run_id="run-a",
        case_time_utc="2026-08-12T00:00:00Z",
        workset_target_gib="31.0",
        calibrated_all_local_mib="31744",
        status="ok",
        prefill_retention="1.0",
        decode_retention="1.0",
    )
    result.update(updates)
    return result


class CapacityPlotTests(unittest.TestCase):
    def fixture(self) -> list[dict[str, str]]:
        return [
            row(
                scheme="intertidal_dma",
                phase="coarse",
                target_offload_pp="0.0",
                page_budget="0",
                logical_working_set_mib="31744",
            ),
            row(
                scheme="intertidal_dma",
                phase="dense",
                target_offload_pp="1.0",
                page_budget="83",
                gross_remote_pp="1.004",
                prefill_retention="0.98",
                decode_retention="0.97",
                logical_working_set_mib="31744",
            ),
            row(
                run_id="run-b",
                case_time_utc="2026-08-12T01:00:00Z",
                scheme="intertidal_dma",
                phase="capacity_frontier",
                workset_target_gib="33.0",
                target_offload_pp="10.6",
                page_budget="875",
                gross_remote_pp="10.59",
                logical_working_set_mib="33792",
                prefill_retention="0.94",
                decode_retention="0.93",
            ),
            row(
                run_id="run-c",
                case_time_utc="2026-08-12T02:00:00Z",
                scheme="intertidal_dma",
                phase="capacity_frontier",
                workset_target_gib="33.0",
                target_offload_pp="10.6",
                page_budget="875",
                gross_remote_pp="10.59",
                logical_working_set_mib="33794",
                prefill_retention="0.90",
                decode_retention="0.89",
            ),
            row(
                scheme="cuda_zero_copy",
                phase="coarse",
                target_offload_pp="0.0",
                page_budget="0",
                logical_working_set_mib="31740",
            ),
            row(
                scheme="cuda_zero_copy",
                phase="dense",
                target_offload_pp="2.0",
                page_budget="165",
                gross_remote_pp="1.997",
                prefill_retention="0.80",
                decode_retention="0.78",
                logical_working_set_mib="31743",
            ),
            row(
                scheme="native_cpu_layers",
                phase="native",
                native_ngl="61",
                native_cpu_model_mib="0",
                native_cuda_model_mib="17000",
                logical_working_set_mib="31750",
            ),
            row(
                scheme="native_cpu_layers",
                phase="native",
                native_ngl="59",
                native_cpu_model_mib="600",
                native_cuda_model_mib="16400",
                logical_working_set_mib="31749",
                prefill_retention="0.55",
                decode_retention="0.52",
            ),
            row(
                scheme="native_cpu_layers",
                phase="capacity_frontier",
                workset_target_gib="33.0",
                native_ngl="58",
                native_cpu_model_mib="900",
                native_cuda_model_mib="16100",
                logical_working_set_mib="33790",
                prefill_retention="0.44",
                decode_retention="0.41",
            ),
            row(
                scheme="intertidal_dma",
                phase="capacity_all_local_control",
                workset_target_gib="34.0",
                target_offload_pp="0.0",
                page_budget="0",
                status="oom",
                prefill_retention="",
                decode_retention="",
            ),
        ]

    def test_build_samples_and_confidence_interval(self) -> None:
        capacity, pure, skipped = plot.build_samples(self.fixture())
        self.assertEqual(len(capacity), 6)
        self.assertEqual(len(pure), 6)
        self.assertEqual(skipped["non-ok or unknown scheme"], 1)

        points = plot.aggregate(capacity, "prefill")["intertidal_dma"]
        repeated = next(point for point in points if point.n == 2)
        self.assertAlmostEqual(repeated.x, (33792 + 33794) / 2 / 1024)
        self.assertAlmostEqual(repeated.mean, 0.92)
        self.assertIsNotNone(repeated.ci_low)
        self.assertIsNotNone(repeated.ci_high)

    def test_svg_has_four_panels_thresholds_and_no_native_line(self) -> None:
        capacity, pure, _ = plot.build_samples(self.fixture())
        svg = plot.render_svg(
            capacity,
            pure,
            title="Smoke",
            sources=[Path("fixture.csv")],
        )
        for panel in (
            "capacity-prefill",
            "pure-prefill",
            "capacity-decode",
            "pure-decode",
        ):
            self.assertIn(f'id="panel-{panel}"', svg)
        self.assertIn("95% target", svg)
        self.assertIn("10% stop", svg)
        self.assertIn('class="ci-whisker scheme-intertidal_dma"', svg)
        self.assertNotIn("series-line scheme-native_cpu_layers", svg)

    def test_cli_writes_svg_and_png_when_rasterizer_is_available(self) -> None:
        if plot.rasterizer() is None:
            self.skipTest("no SVG rasterizer installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "fixture.csv"
            with input_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(self.fixture())
            stem = root / "chart"
            self.assertEqual(
                plot.main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(stem),
                        "--png-scale",
                        "0.25",
                    ]
                ),
                0,
            )
            self.assertGreater(stem.with_suffix(".svg").stat().st_size, 1000)
            png = stem.with_suffix(".png")
            self.assertGreater(png.stat().st_size, 1000)
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
