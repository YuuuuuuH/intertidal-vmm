#!/usr/bin/env python3
"""Plot Gemma capacity-sweep retention without scientific Python packages.

The input is the append-only CSV produced by ``gemma_capacity_sweep.py``.  A
single four-panel figure is written as SVG and, unless ``--svg-only`` is used,
rasterized to PNG with macOS ``sips``, ImageMagick, librsvg, or Inkscape.

The left column uses the *measured* logical working set (process VRAM plus the
reported remote-weight saving).  The right column is the 31 GiB pure-offload
curve and uses the realized gross remote percentage.  Native llama.cpp layer
offload is intentionally rendered as unconnected markers: whole-layer points
must not look like a synthetic 0.1-percentage-point curve.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


WIDTH = 1600
HEIGHT = 1040

SCHEME_ORDER = (
    "intertidal_dma",
    "cuda_zero_copy",
    "native_cpu_layers",
)
SCHEME_LABEL = {
    "intertidal_dma": "Intertidal DMA staging",
    "cuda_zero_copy": "CUDA zero-copy",
    "native_cpu_layers": "Native CPU layers",
}
SCHEME_COLOR = {
    "intertidal_dma": "#0072b2",
    "cuda_zero_copy": "#d55e00",
    "native_cpu_layers": "#009e73",
}
SCHEME_MARKER = {
    "intertidal_dma": "circle",
    "cuda_zero_copy": "square",
    "native_cpu_layers": "triangle",
}

PURE_PHASES = {"pure_curve", "coarse", "dense", "native"}


@dataclass(frozen=True)
class Sample:
    scheme: str
    case_key: tuple[str, ...]
    x: float
    prefill: float
    decode: float


@dataclass(frozen=True)
class Point:
    scheme: str
    x: float
    mean: float
    ci_low: float | None
    ci_high: float | None
    n: int


@dataclass(frozen=True)
class Panel:
    panel_id: str
    letter: str
    title: str
    x_title: str
    samples: tuple[Sample, ...]
    metric: str


def number(row: Mapping[str, str], field: str) -> float | None:
    raw = row.get(field, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_rows(paths: Sequence[Path]) -> tuple[list[dict[str, str]], int]:
    """Read inputs and discard byte-for-byte duplicate benchmark records."""

    rows: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    duplicate_count = 0
    required = {"scheme", "phase", "status", "prefill_retention", "decode_retention"}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: missing CSV fields {sorted(missing)}")
            for row in reader:
                signature = tuple(sorted((key, value or "") for key, value in row.items()))
                if signature in seen:
                    duplicate_count += 1
                    continue
                seen.add(signature)
                rows.append(dict(row))
    return rows, duplicate_count


def control_key(row: Mapping[str, str]) -> tuple[str, ...]:
    scheme = row.get("scheme", "")
    if scheme == "native_cpu_layers":
        return ("ngl", row.get("native_ngl", ""))
    return (
        "pp",
        row.get("target_offload_pp", ""),
        "pages",
        row.get("page_budget", ""),
    )


def is_hybrid_all_local(row: Mapping[str, str]) -> bool:
    if row.get("scheme") == "native_cpu_layers":
        return False
    target = number(row, "target_offload_pp")
    pages = number(row, "page_budget")
    return (target is not None and target == 0) or (pages is not None and pages == 0)


def logical_workset_gib(row: Mapping[str, str]) -> float | None:
    measured = number(row, "logical_working_set_mib")
    if measured is not None and measured > 0:
        return measured / 1024.0

    # Old/partial CSVs may predate process telemetry.  Keep them plottable, but
    # prefer the calibrated all-local footprint over the nominal target.
    calibrated = number(row, "calibrated_all_local_mib")
    if calibrated is not None and calibrated > 0:
        return calibrated / 1024.0
    target = number(row, "workset_target_gib")
    return target if target is not None and target > 0 else None


def realized_remote_pp(row: Mapping[str, str]) -> float | None:
    gross = number(row, "gross_remote_pp")
    if gross is not None and gross >= 0:
        return gross

    scheme = row.get("scheme", "")
    if scheme == "native_cpu_layers":
        cpu = number(row, "native_cpu_model_mib")
        cuda = number(row, "native_cuda_model_mib")
        if cpu is not None and cuda is not None and cpu >= 0 and cuda >= 0 and cpu + cuda > 0:
            return 100.0 * cpu / (cpu + cuda)
        return None

    # A zero-page hybrid control has exactly zero remote bytes even though the
    # writer leaves gross_remote_pp blank for zero-valued accounting fields.
    if is_hybrid_all_local(row):
        return 0.0
    return None


def _native_local_ngl(rows: Iterable[Mapping[str, str]]) -> float | None:
    values = [
        value
        for row in rows
        if row.get("scheme") == "native_cpu_layers"
        and row.get("phase", "") in PURE_PHASES
        and (value := number(row, "native_ngl")) is not None
    ]
    return max(values) if values else None


def build_samples(rows: Sequence[dict[str, str]]) -> tuple[list[Sample], list[Sample], dict[str, int]]:
    """Return capacity and pure-curve samples plus skip accounting."""

    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("scheme") in SCHEME_ORDER
    ]
    native_local_ngl = _native_local_ngl(ok_rows)
    worksets = [number(row, "workset_target_gib") for row in ok_rows]
    minimum_workset = min((value for value in worksets if value is not None), default=None)

    capacity: list[Sample] = []
    pure: list[Sample] = []
    skipped = defaultdict(int)
    for row in ok_rows:
        scheme = row["scheme"]
        prefill = number(row, "prefill_retention")
        decode = number(row, "decode_retention")
        if prefill is None or decode is None or prefill < 0 or decode < 0:
            skipped["missing retention"] += 1
            continue

        phase = row.get("phase", "")
        workset = number(row, "workset_target_gib")
        control = control_key(row)

        is_capacity = phase.startswith("capacity_")
        # The adaptive runner reuses the 31 GiB pure-curve control rather than
        # appending the same case under capacity_frontier.  Add just that true
        # all-local point to the capacity panels, not the whole 31 GiB curve.
        if not is_capacity and minimum_workset is not None and workset == minimum_workset:
            if scheme == "native_cpu_layers":
                ngl = number(row, "native_ngl")
                is_capacity = ngl is not None and ngl == native_local_ngl
            else:
                is_capacity = is_hybrid_all_local(row)

        if is_capacity:
            x = logical_workset_gib(row)
            if x is None:
                skipped["missing logical workset"] += 1
            else:
                case = (
                    scheme,
                    "capacity",
                    row.get("workset_target_gib", ""),
                    *control,
                )
                capacity.append(Sample(scheme, case, x, prefill, decode))

        if phase in PURE_PHASES:
            x = realized_remote_pp(row)
            if x is None:
                skipped["missing realized remote percent"] += 1
            else:
                case = (scheme, "pure", *control)
                pure.append(Sample(scheme, case, x, prefill, decode))

    skipped["non-ok or unknown scheme"] = len(rows) - len(ok_rows)
    return capacity, pure, dict(skipped)


def t_critical_95(degrees_of_freedom: int) -> float:
    # Two-sided Student-t critical values; normal approximation past 30 df.
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    return table.get(degrees_of_freedom, 1.960)


def aggregate(samples: Sequence[Sample], metric: str) -> dict[str, list[Point]]:
    grouped: dict[tuple[str, tuple[str, ...]], list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.scheme, sample.case_key)].append(sample)

    result: dict[str, list[Point]] = {scheme: [] for scheme in SCHEME_ORDER}
    for (scheme, _), repeats in grouped.items():
        values = [getattr(sample, metric) for sample in repeats]
        xs = [sample.x for sample in repeats]
        mean = statistics.fmean(values)
        x = statistics.fmean(xs)
        low: float | None = None
        high: float | None = None
        if len(values) >= 2:
            sem = statistics.stdev(values) / math.sqrt(len(values))
            half = t_critical_95(len(values) - 1) * sem
            low = max(0.0, mean - half)
            high = mean + half
        result[scheme].append(Point(scheme, x, mean, low, high, len(values)))
    for points in result.values():
        points.sort(key=lambda point: point.x)
    return result


def _nice_step(span: float, target_ticks: int = 6) -> float:
    if not math.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(1, target_ticks)
    power = 10.0 ** math.floor(math.log10(raw))
    fraction = raw / power
    if fraction <= 1:
        nice = 1.0
    elif fraction <= 2:
        nice = 2.0
    elif fraction <= 2.5:
        nice = 2.5
    elif fraction <= 5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * power


def nice_domain(values: Sequence[float], *, anchor_zero: bool = False) -> tuple[float, float, list[float]]:
    if not values:
        return (0.0, 1.0, [0.0, 0.5, 1.0])
    minimum = min(values)
    maximum = max(values)
    if anchor_zero and minimum >= 0:
        minimum = 0.0
    if math.isclose(minimum, maximum):
        pad = max(0.5, abs(minimum) * 0.05)
        minimum -= pad
        maximum += pad
        if anchor_zero and minimum < 0:
            minimum = 0.0
    else:
        pad = (maximum - minimum) * 0.04
        minimum -= pad
        maximum += pad
        if anchor_zero and minimum < 0:
            minimum = 0.0
    step = _nice_step(maximum - minimum)
    lo = 0.0 if anchor_zero and minimum >= 0 else math.floor(minimum / step) * step
    hi = math.ceil(maximum / step) * step
    if hi <= lo:
        hi = lo + step
    count = int(round((hi - lo) / step))
    ticks = [lo + index * step for index in range(count + 1)]
    return lo, hi, ticks


def fmt_tick(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def marker_svg(shape: str, x: float, y: float, color: str, size: float = 5.0) -> str:
    common = f'fill="{color}" stroke="#ffffff" stroke-width="1.2"'
    if shape == "square":
        return (
            f'<rect x="{x-size:.2f}" y="{y-size:.2f}" width="{2*size:.2f}" '
            f'height="{2*size:.2f}" {common}/>'
        )
    if shape == "triangle":
        points = f"{x:.2f},{y-size-1:.2f} {x-size-1:.2f},{y+size:.2f} {x+size+1:.2f},{y+size:.2f}"
        return f'<polygon points="{points}" {common}/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" {common}/>'


def render_panel(panel: Panel, x0: float, y0: float, width: float, height: float) -> str:
    left, right, top, bottom = 76.0, 20.0, 48.0, 62.0
    px0, px1 = x0 + left, x0 + width - right
    py0, py1 = y0 + top, y0 + height - bottom
    plot_width, plot_height = px1 - px0, py1 - py0
    series = aggregate(panel.samples, panel.metric)
    all_points = [point for points in series.values() for point in points]
    x_values = [point.x for point in all_points]
    x_min, x_max, x_ticks = nice_domain(
        x_values,
        anchor_zero=panel.panel_id.startswith("pure-"),
    )
    y_values = [0.10, 0.95, 1.0]
    for point in all_points:
        y_values.extend(
            value
            for value in (point.mean, point.ci_low, point.ci_high)
            if value is not None
        )
    y_max = max(1.05, max(y_values) * 1.04)
    y_step = 0.10 if y_max <= 1.21 else _nice_step(y_max, 7)
    y_max = math.ceil(y_max / y_step) * y_step
    y_ticks = [index * y_step for index in range(int(round(y_max / y_step)) + 1)]
    # Make the two decision thresholds explicit ticks even when the regular
    # grid would omit them.
    y_ticks = sorted({*y_ticks, 0.10, 0.95, 1.0})

    def sx(value: float) -> float:
        return px0 + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return py1 - value / y_max * plot_height

    clip_id = f"clip-{panel.panel_id}"
    out = [
        f'<g id="panel-{esc(panel.panel_id)}">',
        f'<text class="panel-letter" x="{x0:.1f}" y="{y0+22:.1f}">{esc(panel.letter)}</text>',
        f'<text class="panel-title" x="{x0+28:.1f}" y="{y0+22:.1f}">{esc(panel.title)}</text>',
        f'<defs><clipPath id="{clip_id}"><rect x="{px0:.2f}" y="{py0:.2f}" width="{plot_width:.2f}" height="{plot_height:.2f}"/></clipPath></defs>',
        f'<rect class="plot-frame" x="{px0:.2f}" y="{py0:.2f}" width="{plot_width:.2f}" height="{plot_height:.2f}"/>',
    ]
    for tick in y_ticks:
        y = sy(tick)
        cls = "grid major" if tick in {0.10, 0.95, 1.0} else "grid"
        out.append(f'<line class="{cls}" x1="{px0:.2f}" y1="{y:.2f}" x2="{px1:.2f}" y2="{y:.2f}"/>')
        out.append(
            f'<text class="tick" text-anchor="end" x="{px0-9:.2f}" y="{y+4:.2f}">{tick*100:.0f}%</text>'
        )
    for tick in x_ticks:
        x = sx(tick)
        out.append(f'<line class="x-grid" x1="{x:.2f}" y1="{py0:.2f}" x2="{x:.2f}" y2="{py1:.2f}"/>')
        out.append(
            f'<text class="tick" text-anchor="middle" x="{x:.2f}" y="{py1+22:.2f}">{esc(fmt_tick(tick))}</text>'
        )

    for threshold, label, line_cls, text_cls in (
        (0.95, "95% target", "target-line", "target-text"),
        (0.10, "10% stop", "stop-line", "stop-text"),
    ):
        y = sy(threshold)
        out.append(
            f'<line class="{line_cls}" x1="{px0:.2f}" y1="{y:.2f}" x2="{px1:.2f}" y2="{y:.2f}"/>'
        )
        out.append(
            f'<text class="threshold-label {text_cls}" text-anchor="end" x="{px1-5:.2f}" y="{y-6:.2f}">{label}</text>'
        )

    out.append(f'<g clip-path="url(#{clip_id})">')
    for scheme in SCHEME_ORDER:
        points = series[scheme]
        if not points:
            continue
        color = SCHEME_COLOR[scheme]
        # Native whole-layer data are categorical residency points.  Deliberately
        # do not connect them: a line would imply unavailable 0.1 pp settings.
        if scheme != "native_cpu_layers" and len(points) >= 2:
            path = " ".join(
                ("M" if index == 0 else "L") + f" {sx(point.x):.2f} {sy(point.mean):.2f}"
                for index, point in enumerate(points)
            )
            out.append(
                f'<path class="series-line scheme-{scheme}" d="{path}" stroke="{color}"/>'
            )
        for point in points:
            x, y = sx(point.x), sy(point.mean)
            if point.ci_low is not None and point.ci_high is not None:
                low, high = sy(point.ci_low), sy(point.ci_high)
                out.extend(
                    [
                        f'<line class="ci-whisker scheme-{scheme}" x1="{x:.2f}" y1="{high:.2f}" x2="{x:.2f}" y2="{low:.2f}" stroke="{color}"/>',
                        f'<line class="ci-cap scheme-{scheme}" x1="{x-5:.2f}" y1="{high:.2f}" x2="{x+5:.2f}" y2="{high:.2f}" stroke="{color}"/>',
                        f'<line class="ci-cap scheme-{scheme}" x1="{x-5:.2f}" y1="{low:.2f}" x2="{x+5:.2f}" y2="{low:.2f}" stroke="{color}"/>',
                    ]
                )
            marker = marker_svg(SCHEME_MARKER[scheme], x, y, color)
            title = (
                f"{SCHEME_LABEL[scheme]}: x={point.x:.3f}, "
                f"retention={point.mean:.2%}, n={point.n}"
            )
            out.append(
                f'<g class="series-marker scheme-{scheme}"><title>{esc(title)}</title>{marker}</g>'
            )
    out.append("</g>")
    if not all_points:
        out.append(
            f'<text class="no-data" text-anchor="middle" x="{(px0+px1)/2:.2f}" y="{(py0+py1)/2:.2f}">No successful samples yet</text>'
        )
    out.extend(
        [
            f'<text class="axis-title" text-anchor="middle" x="{(px0+px1)/2:.2f}" y="{y0+height-12:.2f}">{esc(panel.x_title)}</text>',
            f'<text class="axis-title" text-anchor="middle" transform="translate({x0+17:.2f},{(py0+py1)/2:.2f}) rotate(-90)">Retention vs. 31 GiB control</text>',
            "</g>",
        ]
    )
    return "\n".join(out)


def render_svg(
    capacity: Sequence[Sample],
    pure: Sequence[Sample],
    *,
    title: str,
    sources: Sequence[Path],
) -> str:
    panels = (
        Panel(
            "capacity-prefill",
            "A",
            "Capacity frontier — prefill",
            "Logical working set (GiB)",
            tuple(capacity),
            "prefill",
        ),
        Panel(
            "pure-prefill",
            "B",
            "31 GiB offload curve — prefill",
            "Gross weights in RAM (%)",
            tuple(pure),
            "prefill",
        ),
        Panel(
            "capacity-decode",
            "C",
            "Capacity frontier — decode",
            "Logical working set (GiB)",
            tuple(capacity),
            "decode",
        ),
        Panel(
            "pure-decode",
            "D",
            "31 GiB offload curve — decode",
            "Gross weights in RAM (%)",
            tuple(pure),
            "decode",
        ),
    )
    legend_x = 735
    legend_parts: list[str] = []
    for index, scheme in enumerate(SCHEME_ORDER):
        x = legend_x + index * 255
        marker = marker_svg(SCHEME_MARKER[scheme], x, 79, SCHEME_COLOR[scheme], 5)
        legend_parts.append(
            f'<g class="legend-item scheme-{scheme}">{marker}<text x="{x+13}" y="84">{esc(SCHEME_LABEL[scheme])}</text></g>'
        )
    source_names = ", ".join(path.name for path in sources)
    description = (
        "Four panels compare prefill and decode retention by logical working set "
        "and realized RAM-offload percentage for three memory schemes."
    )
    css = """
    .figure-bg { fill: #ffffff; }
    text { fill: #222222; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; font-size: 13px; }
    .main-title { font-size: 25px; font-weight: 600; }
    .subtitle, .note { fill: #555555; font-size: 12px; }
    .panel-title { font-size: 16px; font-weight: 600; }
    .panel-letter { font-size: 17px; font-weight: 700; }
    .plot-frame { fill: #ffffff; stroke: #777777; stroke-width: 1; }
    .grid, .x-grid { stroke: #e7e7e7; stroke-width: 1; }
    .grid.major { stroke: #d0d0d0; }
    .tick { fill: #454545; font-size: 12px; }
    .axis-title { font-size: 13px; font-weight: 600; }
    .series-line { fill: none; stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; }
    .ci-whisker, .ci-cap { stroke-width: 1.5; opacity: 0.72; }
    .target-line { stroke: #555555; stroke-width: 1.6; stroke-dasharray: 8 5; }
    .stop-line { stroke: #b2182b; stroke-width: 1.5; stroke-dasharray: 3 4; }
    .target-text { fill: #444444; }
    .stop-text { fill: #9b1527; }
    .threshold-label { font-size: 11px; font-weight: 600; }
    .legend-item text { font-size: 12px; }
    .no-data { fill: #777777; font-size: 14px; }
    """
    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="figure-title figure-desc">',
        f'<title id="figure-title">{esc(title)}</title>',
        f'<desc id="figure-desc">{esc(description)}</desc>',
        f"<style>{css}</style>",
        f'<rect class="figure-bg" width="{WIDTH}" height="{HEIGHT}"/>',
        f'<text class="main-title" x="74" y="39">{esc(title)}</text>',
        '<text class="subtitle" x="74" y="62">Measured points only; whiskers are 95% Student-t confidence intervals when the same case has duplicate rows.</text>',
        *legend_parts,
        render_panel(panels[0], 55, 105, 735, 410),
        render_panel(panels[1], 815, 105, 735, 410),
        render_panel(panels[2], 55, 560, 735, 410),
        render_panel(panels[3], 815, 560, 735, 410),
        '<text class="note" x="74" y="1011">Native: measured markers only; x = loader CPU / (CPU + CUDA) model buffers. Hybrid x = gross_remote_pp. No interpolation.</text>',
        f'<text class="note" text-anchor="end" x="1526" y="1011">Source: {esc(source_names)}</text>',
        "</svg>",
    ]
    return "\n".join(body) + "\n"


def write_svg(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".svg"
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def rasterizer() -> tuple[str, str] | None:
    for executable, kind in (
        ("rsvg-convert", "rsvg"),
        ("inkscape", "inkscape"),
        # macOS ships an SVG-capable CoreGraphics rasterizer.  Prefer it to a
        # Homebrew ImageMagick install that may have no configured fonts.
        ("sips", "sips"),
        ("magick", "magick"),
        ("convert", "convert"),
    ):
        path = shutil.which(executable)
        if path:
            return path, kind
    return None


def write_png(svg_path: Path, png_path: Path, scale: float) -> None:
    if scale <= 0:
        raise ValueError("--png-scale must be positive")
    selected = rasterizer()
    if selected is None:
        raise RuntimeError(
            "PNG export needs ImageMagick (`magick`), librsvg (`rsvg-convert`), "
            "or Inkscape; SVG was still written successfully"
        )
    executable, kind = selected
    pixel_width = max(1, round(WIDTH * scale))
    pixel_height = max(1, round(HEIGHT * scale))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = png_path.with_name(f".{png_path.name}.tmp.png")
    if kind == "rsvg":
        command = [
            executable,
            "--width",
            str(pixel_width),
            "--height",
            str(pixel_height),
            "--output",
            str(temporary),
            str(svg_path),
        ]
    elif kind == "inkscape":
        command = [
            executable,
            str(svg_path),
            f"--export-filename={temporary}",
            f"--export-width={pixel_width}",
            f"--export-height={pixel_height}",
        ]
    elif kind == "sips":
        command = [
            executable,
            "-s",
            "format",
            "png",
            "-z",
            str(pixel_height),
            str(pixel_width),
            str(svg_path),
            "--out",
            str(temporary),
        ]
    else:
        command = [
            executable,
            "-background",
            "white",
            str(svg_path),
            "-alpha",
            "remove",
            "-resize",
            f"{pixel_width}x{pixel_height}!",
            str(temporary),
        ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        temporary.replace(png_path)
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        detail = exc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"PNG rasterizer failed: {detail}") from exc


def output_paths(stem: Path) -> tuple[Path, Path]:
    if stem.suffix.lower() in {".svg", ".png"}:
        stem = stem.with_suffix("")
    return stem.with_suffix(".svg"), stem.with_suffix(".png")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="one or more gemma_capacity_sweep.py CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output stem (writes STEM.svg and STEM.png)",
    )
    parser.add_argument(
        "--title",
        default="Gemma-4-31B: capacity expansion vs. performance",
    )
    parser.add_argument(
        "--png-scale",
        type=float,
        default=1.5,
        help="PNG dimensions relative to the 1600x1040 SVG (default: 1.5)",
    )
    parser.add_argument(
        "--svg-only",
        action="store_true",
        help="skip PNG rasterization",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, duplicates = read_rows(args.input)
    capacity, pure, skipped = build_samples(rows)
    svg_path, png_path = output_paths(args.output)
    svg = render_svg(capacity, pure, title=args.title, sources=args.input)
    write_svg(svg_path, svg)
    if not args.svg_only:
        write_png(svg_path, png_path, args.png_scale)

    plotted = len(capacity) + len(pure)
    print(
        f"plotted {plotted} samples from {len(rows)} unique rows "
        f"({duplicates} exact duplicates ignored)"
    )
    for reason, count in sorted(skipped.items()):
        if count:
            print(f"skipped {count}: {reason}")
    print(svg_path)
    if not args.svg_only:
        print(png_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
