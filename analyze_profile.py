#!/usr/bin/env python3
"""Analyze ``intertidal-profile-v1`` JSONL sidecars.

The sweep runners intentionally keep profiling data out of their stable result
CSV schema.  This tool turns the append-only profiling sidecar into three local
artifacts:

* one machine-readable summary CSV row per executed case;
* a Markdown report with formulas, caveats, and likely bottlenecks;
* a dependency-free SVG overview of bandwidth, overlap, and exposed wait.

PCIe utilization is always relative to a measured pinned H2D ceiling.  Supply a
single ceiling with ``--h2d-ceiling-gbps`` or a size sweep in calibration JSONL.
When size-labelled calibration points bracket a copy, the ceiling is
interpolated in log2(transfer bytes).  Heterogeneous layer ceilings are folded
with sampled-byte harmonic weighting; an explicit ceiling can cover sizes
outside the calibrated range.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROTOCOL = "intertidal-profile-v1"
CSV_COLUMNS = [
    "source",
    "line",
    "run_id",
    "status",
    "profile_mode",
    "scheme",
    "phase",
    "profile",
    "probe_kind",
    "control",
    "workset_gib",
    "context_tokens",
    "target_offload_pp",
    "page_budget",
    "native_ngl",
    "elapsed_s",
    "dma_count",
    "dma_bytes",
    "average_dma_bytes",
    "timed_dma_bytes",
    "dma_copy_total_ms",
    "dma_effective_gbps_decimal",
    "h2d_ceiling_gbps_decimal",
    "h2d_calibration_bytes",
    "h2d_ceiling_source",
    "h2d_calibration_method",
    "dma_pcie_utilization_pct",
    "h2d_sustained_ceiling_gbps_decimal",
    "h2d_sustained_ceiling_source",
    "dma_sustained_utilization_pct",
    "case_wall_remote_gbps_decimal",
    "backend_window_remote_gbps_decimal",
    "backend_work_ms",
    "backend_wait_ms",
    "backend_window_ms",
    "exposed_wait_pct",
    "wait_per_dma_us",
    "overlap_pct",
    "prefetch_hit_pct",
    "queue_total_ms",
    "copy_p50_ms",
    "copy_p95_ms",
    "kick_to_ready_p95_ms",
    "wait_p95_ms",
    "wait_max_ms",
    "zero_copy_touches",
    "zero_copy_bytes",
    "telemetry_pcie_rx_mean_gbps_decimal",
    "telemetry_pcie_rx_peak_gbps_decimal",
    "telemetry_rx_peak_utilization_pct",
    "telemetry_pcie_tx_peak_gbps_decimal",
    "gpu_compute_util_mean_pct",
    "gpu_memory_util_mean_pct",
    "gpu_graphics_clock_min_mhz",
    "process_cpu_max_pct",
    "process_major_faults_delta",
    "host_memory_psi_avg10_max_pct",
    "timing_is_sampled",
    "backend_dropped",
    "eval_count",
    "slowest_eval",
    "slowest_eval_wait_ms",
    "slowest_layer",
    "slowest_layer_wait_ms",
    "slowest_layers",
    "diagnosis",
]


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> int | None:
    parsed = _number(value)
    return int(parsed) if parsed is not None else None


def _dig(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _fmt(value: Any, digits: int = 2, missing: str = "-") -> str:
    parsed = _number(value)
    if parsed is None:
        return missing
    return f"{parsed:.{digits}f}"


def _percent(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def _mib_s_to_gbps(value: Any) -> float | None:
    parsed = _number(value)
    return parsed * 1_048_576.0 / 1_000_000_000.0 if parsed is not None else None


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record is not an object")
            records.append((line_number, value))
    return records


@dataclass(frozen=True)
class CalibrationPoint:
    gbps: float
    transfer_bytes: int | None
    source: str
    metric: str = "single_p50"


@dataclass(frozen=True)
class CalibrationChoice:
    gbps: float
    transfer_bytes: int | None
    source: str
    sustained_gbps: float | None = None
    sustained_source: str | None = None


class H2DCalibration:
    def __init__(
        self,
        points: Sequence[CalibrationPoint],
        *,
        fallback_gbps: float | None = None,
    ):
        if fallback_gbps is not None and (not math.isfinite(fallback_gbps) or fallback_gbps <= 0):
            raise ValueError("--h2d-ceiling-gbps must be a positive finite number")
        if not points and fallback_gbps is None:
            raise ValueError("calibration contains no positive H2D GB/s measurements")
        self.points = list(points)
        self.fallback_gbps = fallback_gbps

    @classmethod
    def fixed(cls, gbps: float) -> "H2DCalibration":
        return cls([], fallback_gbps=gbps)

    def _choose_metric(
        self, metric: str, transfer_bytes: float | None
    ) -> tuple[float, int | None, str] | None:
        points = [point for point in self.points if point.metric == metric]
        if not points:
            return None
        sized = [point for point in points if point.transfer_bytes and point.transfer_bytes > 0]
        unsized = [point for point in points if point.transfer_bytes is None]

        grouped: dict[int, list[CalibrationPoint]] = {}
        for point in sized:
            assert point.transfer_bytes is not None
            grouped.setdefault(point.transfer_bytes, []).append(point)

        def group_value(size: int) -> tuple[float, str]:
            members = grouped[size]
            return (
                statistics.median(point.gbps for point in members),
                "; ".join(sorted({point.source for point in members})),
            )

        if grouped and transfer_bytes is not None and transfer_bytes > 0:
            sizes = sorted(grouped)
            if transfer_bytes < sizes[0] and metric == "single_p50" and self.fallback_gbps:
                return None
            if transfer_bytes > sizes[-1] and metric == "single_p50" and self.fallback_gbps:
                return None
            if transfer_bytes <= sizes[0]:
                value, source = group_value(sizes[0])
                return value, sizes[0], source
            if transfer_bytes >= sizes[-1]:
                value, source = group_value(sizes[-1])
                return value, sizes[-1], source
            for lower, upper in zip(sizes, sizes[1:]):
                if transfer_bytes == lower:
                    value, source = group_value(lower)
                    return value, lower, source
                if lower < transfer_bytes < upper:
                    lower_value, lower_source = group_value(lower)
                    upper_value, upper_source = group_value(upper)
                    fraction = math.log2(transfer_bytes / lower) / math.log2(upper / lower)
                    interpolated = lower_value + fraction * (upper_value - lower_value)
                    return (
                        interpolated,
                        int(round(transfer_bytes)),
                        f"log-size interpolation {lower_source} <-> {upper_source}",
                    )

        if unsized:
            return (
                statistics.median(point.gbps for point in unsized),
                None,
                "; ".join(sorted({point.source for point in unsized})),
            )
        if grouped:
            # With no case transfer size, the largest calibration point is the
            # least latency-dominated proxy for the link ceiling.
            size = max(grouped)
            value, source = group_value(size)
            return value, size, source
        return None

    def choose(self, transfer_bytes: float | None) -> CalibrationChoice:
        single = self._choose_metric("single_p50", transfer_bytes)
        if single is None:
            if self.fallback_gbps is None:
                raise ValueError("calibration contains no single-copy H2D ceiling")
            single = (self.fallback_gbps, None, "command line fallback")
        sustained = self._choose_metric("sustained", transfer_bytes)
        return CalibrationChoice(
            single[0],
            single[1],
            single[2],
            sustained[0] if sustained is not None else None,
            sustained[2] if sustained is not None else None,
        )


_SINGLE_BANDWIDTH_KEYS = (
    "h2d_ceiling_gbps",
    # pcie_h2d_calibrate.cu emits both figures.  A backend DMA event measures
    # one copy, so its like-for-like denominator is the same-size single-copy
    # median; sustained train bandwidth remains a fallback for other tools.
    "single_p50_gbps_decimal",
    "median_gbps_decimal",
    "median_gbps",
    "p50_gbps_decimal",
    "p50_gbps",
    "effective_gbps_decimal",
    "effective_gbps",
    "gbps_decimal",
    "gbps",
    "bandwidth_gbps",
)
_SUSTAINED_BANDWIDTH_KEYS = (
    "sustained_gbps_decimal",
    "copy_train_gbps_decimal",
    "train_gbps_decimal",
)
_SIZE_KEYS = ("transfer_bytes", "size_bytes", "bytes")


def _h2d_direction(value: Any) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"h2d", "host_to_device", "host2device", "cpu_to_gpu"}


def _calibration_points(
    value: Any,
    *,
    source: str,
    inherited_h2d: bool = False,
) -> Iterable[CalibrationPoint]:
    if isinstance(value, list):
        for item in value:
            yield from _calibration_points(item, source=source, inherited_h2d=inherited_h2d)
        return
    if not isinstance(value, dict):
        return

    direction_value = value.get("direction", value.get("copy_direction"))
    is_h2d = _h2d_direction(direction_value) if direction_value is not None else inherited_h2d
    # A directly named ceiling is unambiguously H2D even in a compact summary.
    if "h2d_ceiling_gbps" in value:
        is_h2d = True
    if is_h2d:
        single_bandwidth = next(
            (
                _number(value.get(key))
                for key in _SINGLE_BANDWIDTH_KEYS
                if _number(value.get(key)) is not None
            ),
            None,
        )
        sustained_bandwidth = next(
            (
                _number(value.get(key))
                for key in _SUSTAINED_BANDWIDTH_KEYS
                if _number(value.get(key)) is not None
            ),
            None,
        )
        transfer_bytes = next(
            (_integer(value.get(key)) for key in _SIZE_KEYS if _integer(value.get(key)) is not None),
            None,
        )
        if single_bandwidth is not None and single_bandwidth > 0:
            yield CalibrationPoint(single_bandwidth, transfer_bytes, source, "single_p50")
        if sustained_bandwidth is not None and sustained_bandwidth > 0:
            yield CalibrationPoint(sustained_bandwidth, transfer_bytes, source, "sustained")

    for key, child in value.items():
        if isinstance(child, (dict, list)):
            child_h2d = is_h2d or str(key).lower() in {"h2d", "host_to_device"}
            yield from _calibration_points(child, source=source, inherited_h2d=child_h2d)


def load_calibration(path: Path, *, fallback_gbps: float | None = None) -> H2DCalibration:
    points: list[CalibrationPoint] = []
    for line_number, record in read_jsonl(path):
        points.extend(
            _calibration_points(record, source=f"{path.name}:{line_number}")
        )
    return H2DCalibration(points, fallback_gbps=fallback_gbps)


def choose_case_calibration(
    calibration: H2DCalibration,
    layers: Sequence[dict[str, Any]],
    *,
    aggregate_average_bytes: float | None,
    timing_is_sampled: bool,
) -> tuple[CalibrationChoice, str]:
    """Build a byte-weighted mixed ceiling for heterogeneous layer copies.

    For layer ``i``, ``B_i`` is selected from calibration using that layer's
    average sampled copy size.  Copy time at the ceiling is ``bytes_i / B_i``,
    so the equivalent aggregate ceiling is the weighted harmonic form
    ``sum(bytes_i) / sum(bytes_i / B_i)``.  This is materially more accurate
    than looking up one ceiling using the aggregate average when layer copy
    sizes differ.
    """

    layer_choices: list[tuple[float, CalibrationChoice]] = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        count = _integer(layer.get("sampled_copy_count")) or 0
        sampled_bytes = _integer(layer.get("sampled_bytes")) or 0
        if count <= 0 or sampled_bytes <= 0:
            # Version-1 sidecars always expose sampled fields.  The fallback
            # keeps early profile prototypes useful only when timings cover
            # every copy; using unsampled totals with sampled effective GB/s
            # would bias the mixed denominator.
            if timing_is_sampled:
                continue
            count = _integer(layer.get("copy_count")) or 0
            sampled_bytes = _integer(layer.get("total_bytes")) or 0
        if count <= 0 or sampled_bytes <= 0:
            continue
        average_bytes = sampled_bytes / count
        layer_choices.append((float(sampled_bytes), calibration.choose(average_bytes)))

    if not layer_choices:
        return calibration.choose(aggregate_average_bytes), "aggregate_average_copy_size"

    total_bytes = sum(weight for weight, _ in layer_choices)
    single_time = sum(weight / choice.gbps for weight, choice in layer_choices)
    mixed_single = total_bytes / single_time
    single_sources = sorted({choice.source for _, choice in layer_choices})

    sustained_values = [choice.sustained_gbps for _, choice in layer_choices]
    if all(value is not None and value > 0 for value in sustained_values):
        sustained_time = sum(
            weight / float(choice.sustained_gbps)
            for weight, choice in layer_choices
        )
        mixed_sustained = total_bytes / sustained_time
        sustained_sources = sorted(
            {
                choice.sustained_source
                for _, choice in layer_choices
                if choice.sustained_source
            }
        )
        sustained_source = (
            f"layer sampled-byte harmonic ({len(layer_choices)} records): "
            + "; ".join(sustained_sources)
        )
    else:
        mixed_sustained = None
        sustained_source = None

    return (
        CalibrationChoice(
            mixed_single,
            None,
            f"layer sampled-byte harmonic ({len(layer_choices)} records): "
            + "; ".join(single_sources),
            mixed_sustained,
            sustained_source,
        ),
        "layer_sampled_bytes_harmonic",
    )


def _stats_value(record: dict[str, Any], path: Sequence[str], statistic: str) -> float | None:
    return _number(_dig(record, *path, statistic))


def _sum_records(records: Any, key: str) -> float | None:
    if not isinstance(records, list):
        return None
    values = [_number(item.get(key)) for item in records if isinstance(item, dict)]
    valid = [value for value in values if value is not None]
    return sum(valid) if valid else None


def _slowest(records: Any, identity: str) -> tuple[int | None, float | None]:
    if not isinstance(records, list):
        return None, None
    candidates: list[tuple[float, int | None]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        wait = _number(record.get("wait_ms"))
        if wait is not None:
            candidates.append((wait, _integer(record.get(identity))))
    if not candidates:
        return None, None
    wait, item_id = max(candidates)
    return item_id, wait


def _case_label(row: dict[str, Any]) -> str:
    pieces = [str(row.get("scheme") or "unknown")]
    control = row.get("control")
    context = row.get("context_tokens")
    if control not in {None, ""}:
        pieces.append(str(control))
    if context not in {None, ""}:
        pieces.append(f"C{context}")
    return " / ".join(pieces)


def _diagnose(row: dict[str, Any]) -> str:
    hints: list[str] = []
    utilization = _number(row.get("dma_pcie_utilization_pct"))
    wait_share = _number(row.get("exposed_wait_pct"))
    overlap = _number(row.get("overlap_pct"))
    average_bytes = _number(row.get("average_dma_bytes"))
    queue_ms = _number(row.get("queue_total_ms"))
    copy_ms = _number(row.get("dma_copy_total_ms"))
    misses = _number(row.get("prefetch_hit_pct"))
    zero_copy_bytes = _number(row.get("zero_copy_bytes")) or 0.0
    dma_bytes = _number(row.get("dma_bytes")) or 0.0

    if dma_bytes <= 0 and zero_copy_bytes > 0:
        hints.append("zero-copy：DMA event 不代表系统内存读，结合 PCIe RX/Nsight")
    elif utilization is None:
        hints.append("缺少可用的 DMA 时序")
    elif utilization >= 85.0 and (wait_share or 0.0) >= 5.0:
        hints.append("PCIe 接近校准上限且等待已暴露")
    elif utilization < 70.0 and average_bytes is not None and average_bytes < 4 * 1024 * 1024:
        hints.append("PCIe 未吃满；优先检查小块传输/启动延迟")
    elif utilization < 70.0:
        hints.append("PCIe 未吃满；检查排队、同步和源内存状态")

    if overlap is not None and overlap < 90.0 and (_number(row.get("backend_wait_ms")) or 0.0) > 0:
        hints.append("预取覆盖不足")
    if queue_ms is not None and copy_ms is not None and copy_ms > 0 and queue_ms > 0.25 * copy_ms:
        hints.append("copy stream 排队偏高")
    if misses is not None and misses < 95.0:
        hints.append("prefetch miss 偏高")
    if (_number(row.get("process_major_faults_delta")) or 0.0) > 0:
        hints.append("运行期间出现 major fault")
    if (_number(row.get("host_memory_psi_avg10_max_pct")) or 0.0) > 0.1:
        hints.append("主机内存 PSI 显示压力")
    if not hints:
        hints.append("未触发启发式告警")
    return "；".join(hints)


def summarize_record(
    record: dict[str, Any],
    *,
    source: Path,
    line_number: int,
    calibration: H2DCalibration,
) -> dict[str, Any]:
    if record.get("protocol") != PROTOCOL:
        raise ValueError(
            f"{source}:{line_number}: expected protocol={PROTOCOL!r}, "
            f"got {record.get('protocol')!r}"
        )
    case = record.get("case_key") if isinstance(record.get("case_key"), dict) else {}
    backend = record.get("backend") if isinstance(record.get("backend"), dict) else {}
    dma = backend.get("dma") if isinstance(backend.get("dma"), dict) else {}
    zero_copy = (
        backend.get("zero_copy") if isinstance(backend.get("zero_copy"), dict) else {}
    )
    layers = backend.get("layers") if isinstance(backend.get("layers"), list) else []
    slowest_layers = (
        backend.get("slowest_layers_by_wait_ms")
        if isinstance(backend.get("slowest_layers_by_wait_ms"), list)
        else layers
    )
    evals = backend.get("evals") if isinstance(backend.get("evals"), list) else []

    dma_count = _integer(dma.get("count")) or 0
    dma_bytes = _integer(dma.get("bytes")) or 0
    sampled_count = _integer(dma.get("sampled_count")) or 0
    sampled_bytes = _integer(dma.get("sampled_bytes")) or 0
    timing_is_sampled = bool(dma.get("timing_is_sampled"))
    average_dma_bytes = dma_bytes / dma_count if dma_count > 0 else None
    sampled_average_dma_bytes = (
        sampled_bytes / sampled_count if sampled_count > 0 and sampled_bytes > 0 else None
    )
    calibration_choice, calibration_method = choose_case_calibration(
        calibration,
        layers,
        aggregate_average_bytes=sampled_average_dma_bytes or average_dma_bytes,
        timing_is_sampled=timing_is_sampled,
    )
    dma_gbps = _number(dma.get("effective_gbps_decimal"))
    dma_utilization = _percent(dma_gbps, calibration_choice.gbps)
    sustained_utilization = _percent(dma_gbps, calibration_choice.sustained_gbps)
    elapsed_s = _number(record.get("elapsed_s"))

    eval_work_ms = _sum_records(evals, "work_ms")
    eval_wait_ms = _sum_records(evals, "wait_ms")
    grouping = evals
    if eval_work_ms is None and eval_wait_ms is None:
        grouping = layers
        eval_work_ms = _sum_records(layers, "work_ms")
        eval_wait_ms = _sum_records(layers, "wait_ms")
    backend_wait_ms = _number(dma.get("wait_total_ms"))
    if backend_wait_ms is None:
        backend_wait_ms = eval_wait_ms
    backend_work_ms = eval_work_ms
    backend_window_ms = None
    if backend_work_ms is not None or backend_wait_ms is not None:
        backend_window_ms = (backend_work_ms or 0.0) + (backend_wait_ms or 0.0)

    timed_dma_bytes = sampled_bytes if timing_is_sampled else dma_bytes
    zero_copy_bytes = _integer(zero_copy.get("bytes")) or 0
    sampled_zero_copy_bytes = _integer(zero_copy.get("sampled_bytes")) or 0
    remote_bytes = dma_bytes if dma_bytes > 0 else zero_copy_bytes
    timed_remote_bytes = (
        timed_dma_bytes
        if dma_bytes > 0
        else (sampled_zero_copy_bytes if sampled_zero_copy_bytes > 0 else zero_copy_bytes)
    )
    case_wall_gbps = (
        remote_bytes / elapsed_s / 1_000_000_000.0
        if elapsed_s is not None and elapsed_s > 0
        else None
    )
    backend_window_gbps = (
        timed_remote_bytes / backend_window_ms / 1_000_000.0
        if backend_window_ms is not None and backend_window_ms > 0
        else None
    )
    exposed_wait_pct = _percent(backend_wait_ms, backend_window_ms)
    wait_per_dma_us = (
        1000.0 * backend_wait_ms / dma_count
        if backend_wait_ms is not None and dma_count > 0
        else None
    )
    queue_total_ms = _sum_records(grouping, "queue_ms")

    slow_layer, slow_layer_wait = _slowest(slowest_layers, "layer")
    slow_eval, slow_eval_wait = _slowest(evals, "eval")
    slowest_text_parts: list[str] = []
    for layer in sorted(
        (item for item in slowest_layers if isinstance(item, dict)),
        key=lambda item: _number(item.get("wait_ms")) or 0.0,
        reverse=True,
    )[:8]:
        layer_id = _integer(layer.get("layer"))
        wait = _number(layer.get("wait_ms"))
        if wait is not None:
            slowest_text_parts.append(f"L{layer_id if layer_id is not None else '?'}:{wait:.3f}ms")

    rx_mean_gbps = _mib_s_to_gbps(_stats_value(record, ("telemetry", "pcie", "rx_mib_s"), "mean"))
    rx_peak_gbps = _mib_s_to_gbps(_stats_value(record, ("telemetry", "pcie", "rx_mib_s"), "max"))
    tx_peak_gbps = _mib_s_to_gbps(_stats_value(record, ("telemetry", "pcie", "tx_mib_s"), "max"))
    psi_paths = (
        ("telemetry", "host", "psi_avg10_pct", "memory_some", "max"),
        ("telemetry", "host", "psi_avg10_pct", "memory_full", "max"),
    )
    psi_values = [_number(_dig(record, *path)) for path in psi_paths]
    psi_valid = [value for value in psi_values if value is not None]

    row: dict[str, Any] = {
        "source": str(source),
        "line": line_number,
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "profile_mode": record.get("profile_mode"),
        "scheme": case.get("scheme"),
        "phase": case.get("phase"),
        "profile": case.get("profile"),
        "probe_kind": case.get("probe_kind"),
        "control": case.get("control"),
        "workset_gib": case.get("workset_gib"),
        "context_tokens": case.get("context_tokens"),
        "target_offload_pp": case.get("target_offload_pp"),
        "page_budget": case.get("page_budget"),
        "native_ngl": case.get("native_ngl"),
        "elapsed_s": elapsed_s,
        "dma_count": dma_count,
        "dma_bytes": dma_bytes,
        "average_dma_bytes": average_dma_bytes,
        "timed_dma_bytes": timed_dma_bytes,
        "dma_copy_total_ms": _number(dma.get("copy_total_ms")),
        "dma_effective_gbps_decimal": dma_gbps,
        "h2d_ceiling_gbps_decimal": calibration_choice.gbps,
        "h2d_calibration_bytes": calibration_choice.transfer_bytes,
        "h2d_ceiling_source": calibration_choice.source,
        "h2d_calibration_method": calibration_method,
        "dma_pcie_utilization_pct": dma_utilization,
        "h2d_sustained_ceiling_gbps_decimal": calibration_choice.sustained_gbps,
        "h2d_sustained_ceiling_source": calibration_choice.sustained_source,
        "dma_sustained_utilization_pct": sustained_utilization,
        "case_wall_remote_gbps_decimal": case_wall_gbps,
        "backend_window_remote_gbps_decimal": backend_window_gbps,
        "backend_work_ms": backend_work_ms,
        "backend_wait_ms": backend_wait_ms,
        "backend_window_ms": backend_window_ms,
        "exposed_wait_pct": exposed_wait_pct,
        "wait_per_dma_us": wait_per_dma_us,
        "overlap_pct": _number(dma.get("overlap_pct", dma.get("hidden_pct"))),
        "prefetch_hit_pct": _number(dma.get("prefetch_hit_pct")),
        "queue_total_ms": queue_total_ms,
        "copy_p50_ms": _number(dma.get("copy_p50_ms")),
        "copy_p95_ms": _number(dma.get("copy_p95_ms")),
        "kick_to_ready_p95_ms": _number(dma.get("kick_to_ready_p95_ms")),
        "wait_p95_ms": _number(dma.get("wait_p95_ms")),
        "wait_max_ms": _number(dma.get("wait_max_ms")),
        "zero_copy_touches": _integer(zero_copy.get("touches")) or 0,
        "zero_copy_bytes": zero_copy_bytes,
        "telemetry_pcie_rx_mean_gbps_decimal": rx_mean_gbps,
        "telemetry_pcie_rx_peak_gbps_decimal": rx_peak_gbps,
        "telemetry_rx_peak_utilization_pct": _percent(rx_peak_gbps, calibration_choice.gbps),
        "telemetry_pcie_tx_peak_gbps_decimal": tx_peak_gbps,
        "gpu_compute_util_mean_pct": _stats_value(record, ("telemetry", "gpu", "compute_util_pct"), "mean"),
        "gpu_memory_util_mean_pct": _stats_value(record, ("telemetry", "gpu", "memory_util_pct"), "mean"),
        "gpu_graphics_clock_min_mhz": _stats_value(record, ("telemetry", "gpu", "graphics_clock_mhz"), "min"),
        "process_cpu_max_pct": _stats_value(record, ("telemetry", "process", "cpu_pct"), "max"),
        "process_major_faults_delta": _number(_dig(record, "telemetry", "process", "major_faults_delta")),
        "host_memory_psi_avg10_max_pct": max(psi_valid) if psi_valid else None,
        "timing_is_sampled": timing_is_sampled,
        "backend_dropped": _integer(backend.get("dropped")) or 0,
        "eval_count": len(evals),
        "slowest_eval": slow_eval,
        "slowest_eval_wait_ms": slow_eval_wait,
        "slowest_layer": slow_layer,
        "slowest_layer_wait_ms": slow_layer_wait,
        "slowest_layers": ";".join(slowest_text_parts),
    }
    row["diagnosis"] = _diagnose(row)
    row["case_label"] = _case_label(row)
    return row


def load_sidecars(paths: Sequence[Path], calibration: H2DCalibration) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, record in read_jsonl(path):
            rows.append(
                summarize_record(
                    record,
                    source=path,
                    line_number=line_number,
                    calibration=calibration,
                )
            )
    if not rows:
        raise ValueError("profiling sidecar contains no records")
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_escape(value: Any) -> str:
    return str(value if value is not None else "-").replace("|", "\\|").replace("\n", " ")


def write_markdown(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    svg_path: Path,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"记录数：{len(rows)}。GB/s 均为十进制（10^9 bytes/s）。",
        "",
        f"![profiling overview]({svg_path.name})",
        "",
        "## 口径",
        "",
        "- 主 `DMA PCIe 利用率 = 后端单次 CUDA event H2D GB/s / 混合 single-copy p50 上限`。每层先按平均 sampled copy 大小查表（点间按 log2(bytes) 插值），再按 `Bmix = sum(bytes) / sum(bytes/B_layer)` 合成。",
        "- 缺少有效 layer sampled bytes 时，才用整次 sampled 平均 copy 大小查一个 ceiling；`calibration` 列明确记录采用的口径。",
        "- `sustained util` 另以连续 copy train 上限为分母，代表 copy stream 长队列时更严格的链路上限；没有该校准时留空。",
        "- `暴露等待占比 = main-stream wait / (sampled work window + main-stream wait)`。",
        "- `overlap` 由后端事件给出；100% 表示 DMA ready 延迟完全被其他工作遮住。",
        "- `case-wall GB/s` 的分母包含进程启动、加载和同步，只用于发现异常，不代表 decode/prefill 阶段带宽。",
        "- `backend-window GB/s` 使用 sampled bytes 与后端 sampled work/wait 窗口；抽样时不要当作整次运行流量。",
        "- `nvidia-smi` PCIe RX 是低频采样，短 burst 可能被漏掉；zero-copy 要以 RX/Nsight 系统内存读为主。",
        "",
        "## 汇总",
        "",
        "| case | status | calibration | DMA GB/s | PCIe util | sustained util | RX peak GB/s | overlap | exposed wait | slowest layer | diagnosis |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        slow = (
            f"L{row['slowest_layer']} / {_fmt(row['slowest_layer_wait_ms'], 3)} ms"
            if row.get("slowest_layer") is not None
            else "-"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(row["case_label"]),
                    _markdown_escape(row.get("status")),
                    _markdown_escape(row.get("h2d_calibration_method")),
                    _fmt(row.get("dma_effective_gbps_decimal")),
                    f"{_fmt(row.get('dma_pcie_utilization_pct'), 1)}%" if row.get("dma_pcie_utilization_pct") is not None else "-",
                    f"{_fmt(row.get('dma_sustained_utilization_pct'), 1)}%" if row.get("dma_sustained_utilization_pct") is not None else "-",
                    _fmt(row.get("telemetry_pcie_rx_peak_gbps_decimal")),
                    f"{_fmt(row.get('overlap_pct'), 1)}%" if row.get("overlap_pct") is not None else "-",
                    f"{_fmt(row.get('exposed_wait_pct'), 1)}%" if row.get("exposed_wait_pct") is not None else "-",
                    slow,
                    _markdown_escape(row.get("diagnosis")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 最慢层",
            "",
            "| case | layers by exposed wait | slowest eval | backend dropped |",
            "|---|---|---:|---:|",
        ]
    )
    for row in rows:
        slow_eval = (
            f"E{row['slowest_eval']} / {_fmt(row['slowest_eval_wait_ms'], 3)} ms"
            if row.get("slowest_eval") is not None
            else "-"
        )
        lines.append(
            f"| {_markdown_escape(row['case_label'])} | "
            f"{_markdown_escape(row.get('slowest_layers') or '-')} | {slow_eval} | "
            f"{row.get('backend_dropped', 0)} |"
        )

    lines.extend(
        [
            "",
            "## 解读顺序",
            "",
            "1. 先看主 `DMA PCIe util`：接近 100% 才能说 staging 已吃到同块大小的单次拷贝上限；再用 `sustained util` 判断离连续传输上限还有多远。",
            "2. 再看 `overlap` 与 `exposed wait`：带宽高但等待也高，说明需要更早预取或减少 remote 比例。",
            "3. 带宽低时看块大小、queue、prefetch miss、CPU major fault/PSI；它们能区分碎片化、排队与主机内存问题。",
            "4. 最后按 layer/eval 定位关键路径，并用短窗口 Nsight Systems/Compute 复核 kernel、GDDR 和系统内存流量。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _svg_polyline(
    values: Sequence[float | None],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    maximum: float,
    color: str,
    label: str,
) -> str:
    if maximum <= 0:
        maximum = 1.0
    count = len(values)
    coordinates: list[tuple[float, float]] = []
    circles: list[str] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        x = left + (index + 0.5) * width / max(1, count)
        y = top + height - min(max(value, 0.0), maximum) / maximum * height
        coordinates.append((x, y))
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.7" fill="{color}"/>')
    polyline = ""
    if coordinates:
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
        polyline = f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
    return (
        polyline
        + "".join(circles)
        + f'<text x="{left + width - 4:.1f}" y="{top + 14:.1f}" text-anchor="end" '
        f'font-size="12" fill="{color}">{html.escape(label)}</text>'
    )


def write_svg(path: Path, rows: Sequence[dict[str, Any]], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(960, min(2200, 260 + len(rows) * 58))
    height = 810
    left, right = 75.0, 30.0
    plot_width = width - left - right
    panels = [(85.0, 175.0), (315.0, 175.0), (545.0, 150.0)]

    dma = [_number(row.get("dma_effective_gbps_decimal")) for row in rows]
    ceiling = [_number(row.get("h2d_ceiling_gbps_decimal")) for row in rows]
    rx_peak = [_number(row.get("telemetry_pcie_rx_peak_gbps_decimal")) for row in rows]
    bandwidth_max = max([value for value in dma + ceiling + rx_peak if value is not None] or [1.0]) * 1.12
    utilization = [_number(row.get("dma_pcie_utilization_pct")) for row in rows]
    overlap = [_number(row.get("overlap_pct")) for row in rows]
    exposed = [_number(row.get("exposed_wait_pct")) for row in rows]
    percent_max = max(
        100.0,
        max([value for value in utilization + overlap + exposed if value is not None] or [100.0])
        * 1.08,
    )
    total_wait = [_number(row.get("backend_wait_ms")) for row in rows]
    slow_wait = [_number(row.get("slowest_layer_wait_ms")) for row in rows]
    wait_max = max([value for value in total_wait + slow_wait if value is not None] or [1.0]) * 1.12

    chunks = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfd"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="sans-serif" font-size="21" fill="#17202a">{html.escape(title)}</text>',
        '<text x="75" y="57" font-family="sans-serif" font-size="12" fill="#5d6d7e">decimal GB/s; utilization uses layer-size-matched pinned H2D calibration</text>',
    ]

    def axes(top: float, panel_height: float, maximum: float, label: str, suffix: str) -> None:
        chunks.append(
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="#ffffff" stroke="#d5d8dc"/>'
        )
        for tick in range(5):
            ratio = tick / 4.0
            y = top + panel_height - ratio * panel_height
            value = ratio * maximum
            chunks.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#edf0f2"/>'
            )
            chunks.append(
                f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#566573">{value:.1f}{suffix}</text>'
            )
        chunks.append(
            f'<text x="18" y="{top + panel_height / 2:.1f}" transform="rotate(-90 18 {top + panel_height / 2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#34495e">{html.escape(label)}</text>'
        )

    top, panel_height = panels[0]
    axes(top, panel_height, bandwidth_max, "bandwidth", "")
    chunks.append(_svg_polyline(ceiling, left=left, top=top, width=plot_width, height=panel_height, maximum=bandwidth_max, color="#7f8c8d", label="calibrated H2D ceiling"))
    chunks.append(_svg_polyline(dma, left=left, top=top, width=plot_width, height=panel_height, maximum=bandwidth_max, color="#0072b2", label="CUDA-event DMA"))
    chunks.append(_svg_polyline(rx_peak, left=left, top=top, width=plot_width, height=panel_height, maximum=bandwidth_max, color="#e69f00", label="nvidia-smi RX peak"))

    top, panel_height = panels[1]
    axes(top, panel_height, percent_max, "percent", "%")
    chunks.append(_svg_polyline(utilization, left=left, top=top, width=plot_width, height=panel_height, maximum=percent_max, color="#009e73", label="PCIe utilization"))
    chunks.append(_svg_polyline(overlap, left=left, top=top, width=plot_width, height=panel_height, maximum=percent_max, color="#0072b2", label="overlap"))
    chunks.append(_svg_polyline(exposed, left=left, top=top, width=plot_width, height=panel_height, maximum=percent_max, color="#d55e00", label="exposed wait"))

    top, panel_height = panels[2]
    axes(top, panel_height, wait_max, "wait (ms)", "")
    chunks.append(_svg_polyline(total_wait, left=left, top=top, width=plot_width, height=panel_height, maximum=wait_max, color="#cc3311", label="total main-stream wait"))
    chunks.append(_svg_polyline(slow_wait, left=left, top=top, width=plot_width, height=panel_height, maximum=wait_max, color="#aa4499", label="slowest layer wait"))

    label_step = max(1, math.ceil(len(rows) / 24))
    for index, row in enumerate(rows):
        if index % label_step != 0 and index != len(rows) - 1:
            continue
        x = left + (index + 0.5) * plot_width / max(1, len(rows))
        label = html.escape(str(row.get("case_label") or index))
        chunks.append(
            f'<text x="{x:.1f}" y="716" transform="rotate(35 {x:.1f} 716)" text-anchor="start" font-family="sans-serif" font-size="10" fill="#566573">{label}</text>'
        )
    chunks.extend(
        [
            f'<text x="{width - 30}" y="790" text-anchor="end" font-family="sans-serif" font-size="10" fill="#7b7d7d">generated by analyze_profile.py</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(chunks), encoding="utf-8")


def output_paths(prefix: Path) -> tuple[Path, Path, Path]:
    # Treat PREFIX literally so ``results/run.profile`` becomes
    # ``run.profile.summary.csv`` instead of losing its final suffix.
    base = str(prefix)
    return Path(base + ".summary.csv"), Path(base + ".report.md"), Path(base + ".report.svg")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", nargs="+", type=Path, help="intertidal-profile-v1 JSONL")
    parser.add_argument(
        "--h2d-ceiling-gbps",
        type=float,
        help="fixed decimal GB/s ceiling, or fallback when calibration lacks single-copy data",
    )
    parser.add_argument(
        "--calibration-jsonl",
        type=Path,
        help="H2D size-sweep calibration JSONL",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="output prefix (default: SIDE-CAR stem + '-analysis')",
    )
    parser.add_argument("--title", default="Intertidal PCIe profiling report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.calibration_jsonl is None and args.h2d_ceiling_gbps is None:
            raise ValueError("provide --calibration-jsonl and/or --h2d-ceiling-gbps")
        calibration = (
            load_calibration(
                args.calibration_jsonl,
                fallback_gbps=args.h2d_ceiling_gbps,
            )
            if args.calibration_jsonl is not None
            else H2DCalibration.fixed(args.h2d_ceiling_gbps)
        )
        rows = load_sidecars(args.sidecar, calibration)
        prefix = args.output_prefix or args.sidecar[0].with_suffix("").with_name(
            args.sidecar[0].stem + "-analysis"
        )
        csv_path, markdown_path, svg_path = output_paths(prefix)
        write_csv(csv_path, rows)
        write_svg(svg_path, rows, title=args.title)
        write_markdown(markdown_path, rows, svg_path=svg_path, title=args.title)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {len(rows)} records")
    print(csv_path)
    print(markdown_path)
    print(svg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
