#!/usr/bin/env python3
"""Resumable fixed-Gemma capacity and weight-offload benchmark orchestrator.

The orchestrator is intentionally local.  A remote benchmark is executed by a
small shell wrapper delivered over SSH stdin; benchmark output and telemetry
stream back immediately, and no result or temporary log is written remotely.

Three paths are supported:

* ``intertidal_dma``: VMM weight pages with double-buffered H2D staging.
* ``cuda_zero_copy``: the same VMM page layout backed directly by host NUMA.
* ``native_cpu_layers``: stock llama.cpp whole-layer ``-ngl`` offload.

Use ``--dry-run`` to validate a manifest and display the exact case plan
without opening an SSH connection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO


SCHEMA_VERSION = 1
MIB = 1 << 20
GIB = 1 << 30

HYBRID_OVERRIDE = (
    r"^blk[.][0-9]+[.](attn_(k|v|q|output)|ffn_(down|gate|up))"
    r"[.]weight$=CUDA_Hybrid"
)

SCHEMES = ("intertidal_dma", "cuda_zero_copy", "native_cpu_layers")
PHASES = ("pure_curve", "capacity_frontier")
TERMINAL_STATUSES = {"ok", "oom", "unsupported", "correctness_fail"}

TELEMETRY_PREFIX = "__GEMMA_SWEEP_GPU__|"
PCIE_TELEMETRY_PREFIX = "__GEMMA_SWEEP_PCIE__|"
SYSTEM_TELEMETRY_PREFIX = "__GEMMA_SWEEP_SYSTEM__|"
EXIT_PREFIX = "__GEMMA_SWEEP_EXIT__|"
BACKEND_PROFILE_PREFIX = "hybrid_profile,"
PROFILE_SIDECAR_SCHEMA_VERSION = 1
PROFILE_MODES = ("off", "light", "deep")
LOGIT_CHECKSUM_RE = re.compile(
    r"^logit_checksum,(prompt|decode),([0-9a-fA-F]{16})"
    r"(?:,nonfinite=([0-9]+))?$"
)

OOM_RE = re.compile(
    r"(?:out of memory|cudaErrorMemoryAllocation|CUDA error 2\b|"
    r"failed to allocate|unable to allocate|cuMemCreate[^\n]*failed|"
    r"memory allocation failed)",
    re.IGNORECASE,
)
UNSUPPORTED_RE = re.compile(
    r"(?:does not support host-NUMA VMM|requires CUDA toolkit 12[.]6|"
    r"not supported|unsupported)",
    re.IGNORECASE,
)
TRANSPORT_ERROR_RE = re.compile(
    r"(?:Permission denied \(publickey,password\)|Permission denied, please try again|"
    r"Connection (?:refused|timed out|closed)|Connection reset by peer|"
    r"Could not resolve hostname|No route to host|Host key verification failed|"
    r"Control socket connect\([^)]*\): No such file or directory|"
    r"kex_exchange_identification:)",
    re.IGNORECASE,
)


CSV_FIELDS = [
    "schema_version",
    "run_id",
    "case_time_utc",
    "manifest_sha256",
    "baseline_source_workset_gib",
    "scheme",
    "phase",
    "workset_target_gib",
    "calibrated_all_local_mib",
    "context_tokens",
    "target_offload_pp",
    "page_budget",
    "native_ngl",
    "status",
    "exit_code",
    "elapsed_s",
    "requested_pages",
    "gross_pages",
    "gross_remote_bytes",
    "gross_remote_mib",
    "gross_remote_pp",
    "staging_bytes",
    "staging_mib",
    "net_saved_bytes",
    "net_saved_mib",
    "net_saved_pp",
    "net_saved_source",
    "all_local_reference_peak_mib",
    "logical_working_set_mib",
    "selected_tensors",
    "selected_layers",
    "native_cpu_model_mib",
    "native_cuda_model_mib",
    "prefill_tok_s",
    "prefill_std_tok_s",
    "decode_tok_s",
    "decode_std_tok_s",
    "prefill_retention",
    "decode_retention",
    "checksum_status",
    "checksum_sequence_json",
    "checksum_baseline_sha256",
    "nonfinite_logits",
    "telemetry_samples",
    "gpu_process_peak_mib",
    "gpu_device_peak_mib",
    "graphics_clock_mean_mhz",
    "graphics_clock_min_mhz",
    "graphics_clock_max_mhz",
    "temperature_mean_c",
    "temperature_max_c",
    "power_mean_w",
    "power_max_w",
    "gpu_util_mean_pct",
    "gpu_util_max_pct",
    "command_json",
    "layout_json",
    "error_tail",
]


@dataclass(frozen=True)
class Case:
    scheme: str
    phase: str
    workset_gib: Decimal
    context_tokens: int
    target_offload_pp: Decimal | None = None
    page_budget: int | None = None
    native_ngl: int | None = None

    def key(self) -> tuple[str, str, str]:
        control = (
            f"ngl:{self.native_ngl}"
            if self.scheme == "native_cpu_layers"
            else f"pp:{canonical_decimal(self.target_offload_pp or Decimal(0))}"
        )
        return self.scheme, canonical_decimal(self.workset_gib), control


@dataclass
class TelemetrySample:
    epoch_ms: int | None
    pid: int | None
    process_mib: float | None
    device_mib: float | None
    clock_mhz: float | None
    temperature_c: float | None
    power_w: float | None
    utilization_pct: float | None
    memory_utilization_pct: float | None = None
    memory_clock_mhz: float | None = None
    pcie_rx_mib_s: float | None = None
    pcie_tx_mib_s: float | None = None
    pcie_link_gen: float | None = None
    pcie_link_width: float | None = None
    pstate: str | None = None
    process_rss_mib: float | None = None
    process_locked_mib: float | None = None
    process_cpu_pct: float | None = None
    process_cpu_ticks: float | None = None
    total_cpu_ticks: float | None = None
    cpu_count: float | None = None
    process_read_mib: float | None = None
    process_write_mib: float | None = None
    process_major_faults: float | None = None
    host_mem_available_mib: float | None = None
    host_swap_free_mib: float | None = None
    psi_cpu_some_pct: float | None = None
    psi_memory_some_pct: float | None = None
    psi_memory_full_pct: float | None = None
    psi_io_some_pct: float | None = None
    psi_io_full_pct: float | None = None


@dataclass(frozen=True)
class BackendProfileEvent:
    event: str
    values: dict[str, str]
    raw: str


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False
    telemetry: list[TelemetrySample] = field(default_factory=list)
    backend_profile: list[BackendProfileEvent] = field(default_factory=list)
    started_epoch_ms: int | None = None
    ended_epoch_ms: int | None = None


@dataclass(frozen=True)
class LogitChecksum:
    label: str
    checksum: str
    nonfinite: int | None

    def identity(self) -> tuple[str, str]:
        return self.label, self.checksum.lower()


@dataclass
class CorrectnessEvaluation:
    status: str
    entries: list[LogitChecksum] = field(default_factory=list)
    baseline: list[tuple[str, str]] | None = None
    baseline_sha256: str = ""
    nonfinite_logits: int | None = None
    error: str = ""


def canonical_decimal(value: Decimal | float | str) -> str:
    dec = value if isinstance(value, Decimal) else Decimal(str(value))
    text = format(dec.normalize(), "f")
    return "0" if text in {"-0", ""} else text


def decimal_range(start: Decimal, stop: Decimal, step: Decimal) -> list[Decimal]:
    if step <= 0:
        raise ValueError("range step must be positive")
    if stop < start:
        raise ValueError("range stop must be >= start")
    count = int(((stop - start) / step).to_integral_value(rounding=ROUND_HALF_UP))
    values = [start + step * index for index in range(count + 1)]
    if values[-1] != stop:
        values.append(stop)
    return values


def profiling_config(
    manifest: dict[str, Any], mode_override: str | None = None
) -> dict[str, Any]:
    """Return a normalized profiling configuration.

    ``light`` samples coarse system state without enabling any CUDA hot-path
    instrumentation.  ``deep`` additionally enables the backend machine
    records and the driver's one-second PCIe dmon stream.  Keeping those two
    levels separate lets throughput sweeps remain representative while a few
    selected cases can be diagnosed in detail.
    """

    raw = manifest.get("profiling", {})
    if mode_override is not None:
        mode = str(mode_override).lower()
    elif bool(raw.get("enabled", False)):
        mode = str(raw.get("mode", "light")).lower()
    else:
        mode = "off"
    if mode not in PROFILE_MODES:
        raise ValueError(f"profiling mode must be one of {PROFILE_MODES}, got {mode!r}")
    sweep_interval = manifest.get("sweep", {}).get("telemetry_interval_s", 0.25)
    default_interval = 0.2 if mode == "deep" else 1.0
    interval = float(
        raw.get(
            f"{mode}_interval_s",
            raw.get("interval_s", default_interval if mode != "off" else sweep_interval),
        )
    )
    if interval <= 0:
        raise ValueError("profiling.interval_s must be positive")
    # Repeated nvidia-smi process launches below 5 Hz measurably disturb short
    # decode tests.  CUDA-event detail, not nvidia-smi polling, provides the
    # fine time resolution in deep mode.
    interval = max(0.20 if mode == "deep" else 1.0, interval)
    return {
        "mode": mode,
        "enabled": mode != "off",
        "interval_s": interval,
        # CUDA timing records are defined over ordinary stream submissions.
        # Deep mode therefore always forces direct execution; light mode can
        # opt into the same path without timing events to measure observer
        # overhead separately from the loss of CUDA Graph replay.
        "force_direct": mode == "deep" or (
            mode == "light" and bool(raw.get("force_direct", False))
        ),
        "pcie_dmon": bool(
            raw.get(f"{mode}_pcie_dmon", raw.get("pcie_dmon", mode == "deep"))
        ),
        "raw_sidecar": bool(raw.get("raw_sidecar", mode == "deep")),
        "backend_enabled": bool(raw.get("backend_enabled", mode == "deep")),
        "backend_sample_every": max(1, int(raw.get("backend_sample_every", 1))),
        "backend_summary_every": max(1, int(raw.get("backend_summary_every", 128))),
        "backend_ring": min(4096, max(8, int(raw.get("backend_ring", 256)))),
        "backend_detail": bool(raw.get("backend_detail", mode == "deep")),
    }


def enable_backend_profile(env: dict[str, str], profile: dict[str, Any]) -> None:
    """Add backend profiler controls to a per-case environment in deep mode."""

    if profile.get("force_direct", False):
        env.setdefault("GGML_CUDA_HYBRID_FORCE_DIRECT", "1")
    if profile["mode"] != "deep" or not profile["backend_enabled"]:
        return
    env.setdefault("GGML_CUDA_HYBRID_PROFILE", "1")
    env.setdefault(
        "GGML_CUDA_HYBRID_PROFILE_SAMPLE_EVERY",
        str(profile["backend_sample_every"]),
    )
    env.setdefault(
        "GGML_CUDA_HYBRID_PROFILE_SUMMARY_EVERY",
        str(profile["backend_summary_every"]),
    )
    env.setdefault("GGML_CUDA_HYBRID_PROFILE_RING", str(profile["backend_ring"]))
    env.setdefault(
        "GGML_CUDA_HYBRID_PROFILE_DETAIL",
        "1" if profile["backend_detail"] else "0",
    )
    # The CUDA-event protocol is defined over direct stream executions.  Make
    # that requirement explicit even though profiling currently forces the
    # same path internally; this also protects the record semantics if graph
    # selection is refactored later.
    env.setdefault("GGML_CUDA_HYBRID_FORCE_DIRECT", "1")


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version must be {SCHEMA_VERSION}, got "
            f"{manifest.get('schema_version')!r}"
        )
    validate_manifest(manifest)
    return manifest, hashlib.sha256(raw).hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> None:
    for section in ("remote", "paths", "model", "benchmark", "worksets", "sweep"):
        if section not in manifest:
            raise ValueError(f"manifest is missing section {section!r}")

    remote = manifest["remote"]
    if remote.get("transport", "ssh") not in {"ssh", "local"}:
        raise ValueError("remote.transport must be 'ssh' or 'local'")
    if remote.get("transport", "ssh") == "ssh" and not remote.get("target"):
        raise ValueError("remote.target is required for SSH transport")

    paths = manifest["paths"]
    for key in ("model", "hybrid_bench", "native_bench"):
        if not paths.get(key):
            raise ValueError(f"paths.{key} is required")

    model = manifest["model"]
    if int(model.get("tensor_bytes", 0)) <= 0:
        raise ValueError("model.tensor_bytes must be positive")
    if int(model.get("page_bytes", 0)) <= 0:
        raise ValueError("model.page_bytes must be positive")

    sweep = manifest["sweep"]
    unknown = set(sweep.get("schemes", SCHEMES)) - set(SCHEMES)
    if unknown:
        raise ValueError(f"unknown schemes: {sorted(unknown)}")
    unknown_phases = set(sweep.get("phases", PHASES)) - set(PHASES)
    if unknown_phases:
        raise ValueError(f"unknown phases: {sorted(unknown_phases)}")
    if Decimal(str(sweep.get("dense_step_pp", "0.1"))) <= 0:
        raise ValueError("sweep.dense_step_pp must be positive")
    if Decimal(str(sweep.get("coarse_step_pp", "1.0"))) <= 0:
        raise ValueError("sweep.coarse_step_pp must be positive")
    threshold = float(sweep.get("stop_retention", 0.10))
    if not 0 < threshold <= 1:
        raise ValueError("sweep.stop_retention must be in (0, 1]")
    if int(sweep.get("stop_consecutive", 2)) < 1:
        raise ValueError("sweep.stop_consecutive must be >= 1")
    pure_workset = Decimal(str(sweep.get("pure_curve_workset_gib", 31)))
    if pure_workset not in workset_values(manifest):
        raise ValueError("sweep.pure_curve_workset_gib must appear in worksets.gib")
    if float(manifest["worksets"].get("default_fit_mib", 0)) <= 0:
        raise ValueError("worksets.default_fit_mib must be positive")

    correctness = manifest.get("correctness", {})
    if correctness.get("enabled", True):
        strict_schemes = set(
            correctness.get(
                "strict_checksum_schemes",
                ["intertidal_dma", "cuda_zero_copy"],
            )
        )
        unknown_strict = strict_schemes - set(SCHEMES)
        if unknown_strict:
            raise ValueError(
                f"correctness.strict_checksum_schemes contains unknown schemes: "
                f"{sorted(unknown_strict)}"
            )
        baseline = manifest_checksum_baseline(manifest)
        if baseline is not None:
            validate_checksum_shape(manifest, baseline, source="manifest baseline")
        if not correctness.get("auto_baseline", True) and baseline is None:
            raise ValueError(
                "correctness requires baseline_checksums when auto_baseline is false"
            )


def workset_values(manifest: dict[str, Any]) -> list[Decimal]:
    spec = manifest["worksets"].get("gib", [31, 32, 33, 34])
    if isinstance(spec, list):
        values = sorted({Decimal(str(item)) for item in spec})
    elif isinstance(spec, dict):
        values = decimal_range(
            Decimal(str(spec["start"])),
            Decimal(str(spec["stop"])),
            Decimal(str(spec["step"])),
        )
    else:
        raise ValueError("worksets.gib must be a list or start/stop/step object")
    if not values:
        raise ValueError("worksets.gib is empty")
    return values


def context_for_workset(manifest: dict[str, Any], workset_gib: Decimal) -> int:
    worksets = manifest["worksets"]
    mapping = worksets.get("context_map", {})
    candidates = [
        canonical_decimal(workset_gib),
        f"{workset_gib:.1f}",
        str(float(workset_gib)),
    ]
    for key in candidates:
        if key in mapping:
            return int(mapping[key])

    calibration = worksets.get("calibration", {})
    intercept_mib = Decimal(str(calibration["intercept_mib"]))
    bytes_per_token = Decimal(str(calibration["kv_bytes_per_token"]))
    quantum = int(calibration.get("context_quantum", 256))
    if bytes_per_token <= 0 or quantum <= 0:
        raise ValueError("KV calibration and context quantum must be positive")

    target_mib = workset_gib * Decimal(1024)
    raw_tokens = (target_mib - intercept_mib) * Decimal(MIB) / bytes_per_token
    rounded_quanta = (raw_tokens / Decimal(quantum)).to_integral_value(
        rounding=ROUND_HALF_UP
    )
    return max(1, int(rounded_quanta) * quantum)


def calibrated_peak_mib(manifest: dict[str, Any], context_tokens: int) -> float:
    calibration = manifest["worksets"].get("calibration", {})
    intercept = float(calibration["intercept_mib"])
    per_token = float(calibration["kv_bytes_per_token"]) / MIB
    return intercept + context_tokens * per_token


def correctness_enabled(manifest: dict[str, Any]) -> bool:
    return bool(manifest.get("correctness", {}).get("enabled", True))


def expected_checksum_labels(manifest: dict[str, Any]) -> list[str]:
    repetitions = int(manifest["benchmark"].get("repetitions", 5))
    per_test = repetitions + (0 if manifest["benchmark"].get("no_warmup", False) else 1)
    return ["prompt"] * per_test + ["decode"] * per_test


def _normalize_checksum(value: object, *, source: str) -> str:
    checksum = str(value).lower()
    if re.fullmatch(r"[0-9a-f]{16}", checksum) is None:
        raise ValueError(f"{source} has invalid 64-bit checksum {value!r}")
    return checksum


def manifest_checksum_baseline(
    manifest: dict[str, Any],
) -> list[tuple[str, str]] | None:
    raw = manifest.get("correctness", {}).get("baseline_checksums")
    if raw is None or raw == {} or raw == []:
        return None

    result: list[tuple[str, str]] = []
    if isinstance(raw, dict):
        unknown = set(raw) - {"prompt", "decode"}
        if unknown:
            raise ValueError(
                f"correctness.baseline_checksums has unknown labels: {sorted(unknown)}"
            )
        for label in ("prompt", "decode"):
            values = raw.get(label, [])
            if not isinstance(values, list):
                raise ValueError(
                    f"correctness.baseline_checksums.{label} must be a list"
                )
            result.extend(
                (label, _normalize_checksum(value, source=f"baseline {label}"))
                for value in values
            )
        return result

    if isinstance(raw, list):
        for index, item in enumerate(raw):
            if not isinstance(item, dict) or set(item) != {"label", "checksum"}:
                raise ValueError(
                    "list-form correctness.baseline_checksums entries must contain "
                    "exactly label and checksum"
                )
            label = str(item["label"])
            if label not in {"prompt", "decode"}:
                raise ValueError(f"baseline entry {index} has invalid label {label!r}")
            result.append(
                (
                    label,
                    _normalize_checksum(
                        item["checksum"], source=f"baseline entry {index}"
                    ),
                )
            )
        return result
    raise ValueError("correctness.baseline_checksums must be an object, list, or null")


def validate_checksum_shape(
    manifest: dict[str, Any],
    identities: Sequence[tuple[str, str]],
    *,
    source: str,
) -> None:
    labels = [label for label, _ in identities]
    expected = expected_checksum_labels(manifest)
    if labels != expected:
        expected_prompt = expected.count("prompt")
        expected_decode = expected.count("decode")
        actual_prompt = labels.count("prompt")
        actual_decode = labels.count("decode")
        raise ValueError(
            f"{source} checksum sequence shape mismatch: expected "
            f"prompt={expected_prompt},decode={expected_decode} in prompt-then-decode "
            f"order; got prompt={actual_prompt},decode={actual_decode},labels={labels}"
        )


def checksum_baseline_sha256(identities: Sequence[tuple[str, str]]) -> str:
    canonical = json.dumps(
        [{"label": label, "checksum": checksum} for label, checksum in identities],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def parse_logit_checksums(text: str) -> list[LogitChecksum]:
    result: list[LogitChecksum] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("logit_checksum,"):
            continue
        match = LOGIT_CHECKSUM_RE.fullmatch(line)
        if match is None:
            # Preserve a sentinel entry so the shape/finite gate fails with a
            # useful diagnostic instead of silently ignoring malformed output.
            result.append(LogitChecksum("malformed", line, None))
            continue
        label, checksum, nonfinite = match.groups()
        result.append(
            LogitChecksum(
                label=label,
                checksum=checksum.lower(),
                nonfinite=int(nonfinite) if nonfinite is not None else None,
            )
        )
    return result


def is_all_local_control(manifest: dict[str, Any], case: Case) -> bool:
    if case.scheme == "native_cpu_layers":
        start = int(
            manifest["sweep"].get(
                "native_ngl_start", manifest["model"].get("gpu_layers", 61)
            )
        )
        return case.native_ngl == start
    return case.target_offload_pp == Decimal(0)


def compare_checksum_sequences(
    expected: Sequence[tuple[str, str]], actual: Sequence[tuple[str, str]]
) -> str:
    if len(expected) != len(actual):
        return f"checksum count mismatch: expected {len(expected)}, got {len(actual)}"
    for index, (wanted, observed) in enumerate(zip(expected, actual)):
        if wanted != observed:
            return (
                f"checksum mismatch at sequence index {index}: "
                f"expected {wanted[0]}={wanted[1]}, got {observed[0]}={observed[1]}"
            )
    return ""


def evaluate_correctness(
    manifest: dict[str, Any],
    case: Case,
    result: CommandResult,
    base_status: str,
    baseline: list[tuple[str, str]] | None,
) -> CorrectnessEvaluation:
    if not correctness_enabled(manifest):
        return CorrectnessEvaluation(status="disabled")
    if base_status != "ok":
        # OOM and unsupported cases do not produce a complete inference and
        # remain capacity outcomes rather than checksum failures.
        return CorrectnessEvaluation(status="not_run")

    config = manifest.get("correctness", {})
    entries = parse_logit_checksums(result.stdout + "\n" + result.stderr)
    identities = [entry.identity() for entry in entries]
    try:
        validate_checksum_shape(manifest, identities, source="case")
    except ValueError as exc:
        return CorrectnessEvaluation(
            status="fail",
            entries=entries,
            baseline=baseline,
            baseline_sha256=checksum_baseline_sha256(baseline) if baseline else "",
            error=str(exc),
        )

    missing_finite = [index for index, entry in enumerate(entries) if entry.nonfinite is None]
    if missing_finite and config.get("require_nonfinite_field", True):
        return CorrectnessEvaluation(
            status="fail",
            entries=entries,
            baseline=baseline,
            baseline_sha256=checksum_baseline_sha256(baseline) if baseline else "",
            error=(
                "checksum lines do not report nonfinite counts at indices "
                f"{missing_finite}; rebuild the benchmark with the finite-logit audit"
            ),
        )

    nonfinite = sum(entry.nonfinite or 0 for entry in entries)
    if nonfinite:
        bad = [
            f"{index}:{entry.label}={entry.nonfinite}"
            for index, entry in enumerate(entries)
            if entry.nonfinite
        ]
        return CorrectnessEvaluation(
            status="fail",
            entries=entries,
            baseline=baseline,
            baseline_sha256=checksum_baseline_sha256(baseline) if baseline else "",
            nonfinite_logits=nonfinite,
            error=f"non-finite logits detected ({', '.join(bad)})",
        )

    strict_schemes = set(
        config.get(
            "strict_checksum_schemes", ["intertidal_dma", "cuda_zero_copy"]
        )
    )
    if case.scheme not in strict_schemes:
        return CorrectnessEvaluation(
            status="finite",
            entries=entries,
            baseline=baseline,
            baseline_sha256=checksum_baseline_sha256(baseline) if baseline else "",
            nonfinite_logits=0,
        )

    if baseline is None:
        if config.get("auto_baseline", True) and is_all_local_control(manifest, case):
            return CorrectnessEvaluation(
                status="baseline_captured",
                entries=entries,
                baseline=identities,
                baseline_sha256=checksum_baseline_sha256(identities),
                nonfinite_logits=0,
            )
        return CorrectnessEvaluation(
            status="fail",
            entries=entries,
            nonfinite_logits=0,
            error=(
                "strict checksum baseline is unavailable; provide "
                "correctness.baseline_checksums or run an all-local control first"
            ),
        )

    mismatch = compare_checksum_sequences(baseline, identities)
    if mismatch:
        return CorrectnessEvaluation(
            status="fail",
            entries=entries,
            baseline=baseline,
            baseline_sha256=checksum_baseline_sha256(baseline),
            nonfinite_logits=0,
            error=mismatch,
        )
    return CorrectnessEvaluation(
        status="match",
        entries=entries,
        baseline=baseline,
        baseline_sha256=checksum_baseline_sha256(baseline),
        nonfinite_logits=0,
    )


def page_budget_for_pp(manifest: dict[str, Any], target_pp: Decimal) -> int:
    model_bytes = Decimal(int(manifest["model"]["tensor_bytes"]))
    page_bytes = Decimal(int(manifest["model"]["page_bytes"]))
    pages = (model_bytes * target_pp / Decimal(100) / page_bytes).to_integral_value(
        rounding=ROUND_HALF_UP
    )
    return max(0, int(pages))


def frontier_required_mib(manifest: dict[str, Any], workset_gib: Decimal) -> float:
    context = context_for_workset(manifest, workset_gib)
    default_fit_mib = float(manifest["worksets"]["default_fit_mib"])
    return max(0.0, calibrated_peak_mib(manifest, context) - default_fit_mib)


def frontier_start_pp(manifest: dict[str, Any], workset_gib: Decimal) -> Decimal:
    """Return a rigorous gross-offload lower bound rounded up to the dense grid.

    Gross bytes upper-bound net VRAM savings, so a point below this estimate
    cannot cover the calibrated overage even with zero staging overhead.
    """

    required_mib = Decimal(str(frontier_required_mib(manifest, workset_gib)))
    model_mib = Decimal(int(manifest["model"]["tensor_bytes"])) / Decimal(MIB)
    step = Decimal(str(manifest["sweep"].get("dense_step_pp", "0.1")))
    raw_pp = required_mib / model_mib * Decimal(100)
    steps = (raw_pp / step).to_integral_value(rounding=ROUND_CEILING)
    return max(Decimal(0), steps * step)


def frontier_pp_targets(manifest: dict[str, Any], workset_gib: Decimal) -> list[Decimal]:
    start = frontier_start_pp(manifest, workset_gib)
    maximum = Decimal(str(manifest["sweep"].get("max_offload_pp", 95)))
    if start > maximum:
        return []
    step = Decimal(str(manifest["sweep"].get("dense_step_pp", "0.1")))
    return decimal_range(start, maximum, step)


def pp_targets(manifest: dict[str, Any], dense: bool) -> list[Decimal]:
    sweep = manifest["sweep"]
    step_key = "dense_step_pp" if dense else "coarse_step_pp"
    return decimal_range(
        Decimal("0"),
        Decimal(str(sweep.get("max_offload_pp", 95))),
        Decimal(str(sweep.get(step_key, "0.1" if dense else "1.0"))),
    )


def build_case(
    manifest: dict[str, Any],
    scheme: str,
    phase: str,
    workset: Decimal,
    *,
    target_pp: Decimal | None = None,
    native_ngl: int | None = None,
) -> Case:
    context = context_for_workset(manifest, workset)
    return Case(
        scheme=scheme,
        phase=phase,
        workset_gib=workset,
        context_tokens=context,
        target_offload_pp=target_pp,
        page_budget=(
            page_budget_for_pp(manifest, target_pp)
            if target_pp is not None and scheme != "native_cpu_layers"
            else None
        ),
        native_ngl=native_ngl,
    )


def common_bench_args(manifest: dict[str, Any], case: Case) -> list[str]:
    bench_cfg = manifest["benchmark"]
    paths = manifest["paths"]
    bench = paths["native_bench"] if case.scheme == "native_cpu_layers" else paths["hybrid_bench"]
    args = [
        str(bench),
        "-m",
        str(paths["model"]),
        "-p",
        str(int(bench_cfg.get("prompt", 512))),
        "-n",
        str(int(bench_cfg.get("generation", 64))),
        "-r",
        str(int(bench_cfg.get("repetitions", 5))),
        "-t",
        str(int(bench_cfg.get("threads", 8))),
        "-b",
        str(int(bench_cfg.get("batch", 512))),
        "-ub",
        str(int(bench_cfg.get("ubatch", 512))),
        "-fa",
        str(bench_cfg.get("flash_attention", "on")),
        "-ctk",
        str(bench_cfg.get("cache_type_k", "f16")),
        "-ctv",
        str(bench_cfg.get("cache_type_v", "f16")),
        "-mmp",
        "1" if bench_cfg.get("mmap", True) else "0",
        "-o",
        "csv",
        str(bench_cfg.get("ctx_arg", "--ctx-size")),
        str(case.context_tokens),
    ]
    if bench_cfg.get("no_warmup", False):
        args.append("--no-warmup")
    for extra in bench_cfg.get("extra_args", []):
        args.append(str(extra))
    return args


def command_for_case(manifest: dict[str, Any], case: Case) -> tuple[dict[str, str], list[str]]:
    env = {str(key): str(value) for key, value in manifest.get("environment", {}).items()}
    if correctness_enabled(manifest):
        # Force the audit on for every successful case.  Strict VMM schemes
        # compare the full sequence with all-local; native CPU-layer runs at
        # least enforce that every emitted logit vector is finite.
        env["LLAMA_BENCH_LOGIT_CHECKSUM"] = "1"
    args = common_bench_args(manifest, case)

    if case.scheme in {"intertidal_dma", "cuda_zero_copy"}:
        assert case.target_offload_pp is not None and case.page_budget is not None
        if case.page_budget > 0:
            env["GGML_CUDA_HYBRID_PAGE_BUDGET"] = str(case.page_budget)
            env["GGML_CUDA_HYBRID_MODE"] = (
                "staging" if case.scheme == "intertidal_dma" else "zero_copy"
            )
            args.extend(
                ["-ot", str(manifest["model"].get("hybrid_override", HYBRID_OVERRIDE))]
            )
    elif case.scheme == "native_cpu_layers":
        assert case.native_ngl is not None
        args.extend(["-ngl", str(case.native_ngl)])
    else:
        raise ValueError(f"unsupported scheme {case.scheme!r}")

    return env, args


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def remote_wrapper(
    env: dict[str, str],
    command: Sequence[str],
    telemetry_interval_s: float,
    gpu_id: str,
    profile: dict[str, Any] | None = None,
) -> str:
    env_words = [f"{key}={value}" for key, value in sorted(env.items())]
    launched = ["env", "-u", "LD_PRELOAD", *env_words, *command]
    profile = profile or {"mode": "off", "enabled": False, "pcie_dmon": False}
    profile_enabled = bool(profile.get("enabled", False))
    interval = max(0.20 if profile_enabled else 0.05, float(telemetry_interval_s))
    system_probe = ""
    pcie_dmon_start = ""
    pcie_dmon_cleanup = ""
    if profile_enabled:
        system_probe = f'''
        gpu_extra=$(nvidia-smi -i {shlex.quote(str(gpu_id))} \\
            --query-gpu=utilization.memory,clocks.current.memory,pstate,pcie.link.gen.current,pcie.link.width.current \\
            --format=csv,noheader,nounits 2>/dev/null | head -n 1)
        IFS=',' read -r mem_util_pct mem_clock_mhz pstate link_gen link_width <<< "$gpu_extra"
        pcie_csv=$(nvidia-smi -i {shlex.quote(str(gpu_id))} \\
            --query-gpu=pcie.rx_util,pcie.tx_util --format=csv,noheader,nounits \\
            2>/dev/null | head -n 1)
        IFS=',' read -r pcie_rx_kib_s pcie_tx_kib_s <<< "$pcie_csv"
        proc_status=$(awk '
            /^VmRSS:/ {{ rss=$2/1024 }} /^VmLck:/ {{ lck=$2/1024 }}
            END {{ printf "%.6f,%.6f", rss+0, lck+0 }}
        ' "/proc/$bench_pid/status" 2>/dev/null)
        IFS=',' read -r proc_rss_mib proc_locked_mib <<< "$proc_status"
        proc_stat=$(awk '{{ print $12 "," ($14+$15) }}' "/proc/$bench_pid/stat" 2>/dev/null)
        IFS=',' read -r proc_major_faults proc_cpu_ticks <<< "$proc_stat"
        cpu_snapshot=$(awk '
            /^cpu / {{ for (i=2;i<=NF;i++) total += $i }}
            /^cpu[0-9]+ / {{ count += 1 }}
            END {{ print total "," count }}
        ' /proc/stat 2>/dev/null)
        IFS=',' read -r total_cpu_ticks cpu_count <<< "$cpu_snapshot"
        proc_io=$(awk '
            /^read_bytes:/ {{ r=$2/1048576 }} /^write_bytes:/ {{ w=$2/1048576 }}
            END {{ printf "%.6f,%.6f", r+0, w+0 }}
        ' "/proc/$bench_pid/io" 2>/dev/null)
        IFS=',' read -r proc_read_mib proc_write_mib <<< "$proc_io"
        host_mem=$(awk '
            /^MemAvailable:/ {{ a=$2/1024 }} /^SwapFree:/ {{ s=$2/1024 }}
            END {{ printf "%.6f,%.6f", a+0, s+0 }}
        ' /proc/meminfo 2>/dev/null)
        IFS=',' read -r mem_available_mib swap_free_mib <<< "$host_mem"
        psi_avg10() {{
            local path="$1" kind="$2"
            awk -v wanted="$kind" '$1 == wanted {{
                for (i=2;i<=NF;i++) if ($i ~ /^avg10=/) {{ split($i,a,"="); print a[2]; exit }}
            }}' "$path" 2>/dev/null
        }}
        printf '{SYSTEM_TELEMETRY_PREFIX}epoch_ms=%s|pid=%s|proc_rss_mib=%s|proc_locked_mib=%s|proc_major_faults=%s|proc_cpu_ticks=%s|total_cpu_ticks=%s|cpu_count=%s|proc_read_mib=%s|proc_write_mib=%s|mem_available_mib=%s|swap_free_mib=%s|psi_cpu_some_pct=%s|psi_memory_some_pct=%s|psi_memory_full_pct=%s|psi_io_some_pct=%s|psi_io_full_pct=%s|memory_util_pct=%s|memory_clock_mhz=%s|pstate=%s|pcie_link_gen=%s|pcie_link_width=%s|pcie_rx_kib_s=%s|pcie_tx_kib_s=%s\\n' \\
            "$(trim "$epoch_ms")" "$bench_pid" "$(trim "$proc_rss_mib")" \\
            "$(trim "$proc_locked_mib")" "$(trim "$proc_major_faults")" \\
            "$(trim "$proc_cpu_ticks")" "$(trim "$total_cpu_ticks")" \\
            "$(trim "$cpu_count")" \\
            "$(trim "$proc_read_mib")" "$(trim "$proc_write_mib")" \\
            "$(trim "$mem_available_mib")" "$(trim "$swap_free_mib")" \\
            "$(psi_avg10 /proc/pressure/cpu some)" \\
            "$(psi_avg10 /proc/pressure/memory some)" \\
            "$(psi_avg10 /proc/pressure/memory full)" \\
            "$(psi_avg10 /proc/pressure/io some)" "$(psi_avg10 /proc/pressure/io full)" \\
            "$(trim "$mem_util_pct")" "$(trim "$mem_clock_mhz")" "$(trim "$pstate")" \\
            "$(trim "$link_gen")" "$(trim "$link_width")" \\
            "$(trim "$pcie_rx_kib_s")" "$(trim "$pcie_tx_kib_s")" >&2
'''
    if profile_enabled and bool(profile.get("pcie_dmon", False)):
        pcie_dmon_start = f'''
pcie_monitor() {{
    trap 'jobs -pr | xargs -r kill 2>/dev/null || true' EXIT INT TERM HUP
    nvidia-smi dmon -i {shlex.quote(str(gpu_id))} -s t -d 1 2>/dev/null | \\
        awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ {{
            cmd="date +%s%3N"; cmd | getline epoch; close(cmd);
            printf "{PCIE_TELEMETRY_PREFIX}%s|%s|%s\\n", epoch, $2, $3;
            fflush();
        }}' >&2
}}
pcie_monitor &
pcie_monitor_pid=$!
'''
        pcie_dmon_cleanup = '''
if [[ -n "$pcie_monitor_pid" ]]; then
    kill "$pcie_monitor_pid" 2>/dev/null || true
    wait "$pcie_monitor_pid" 2>/dev/null || true
    pcie_monitor_pid=''
fi
'''
    return f"""#!/usr/bin/env bash
set +e
# The machine exports a site-specific CUDA interposer through LD_PRELOAD.
# Remove it for the entire wrapper (including nvidia-smi/awk/sleep), then add
# only the explicit per-case environment below.  Otherwise telemetry helpers
# can emit interposer records and perturb the memory measurements.
unset LD_PRELOAD
ulimit -c 0
bench_pid=''
monitor_pid=''
pcie_monitor_pid=''
cleanup() {{
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
    if [[ -n "$pcie_monitor_pid" ]]; then kill "$pcie_monitor_pid" 2>/dev/null || true; fi
    if [[ -n "$bench_pid" ]]; then kill "$bench_pid" 2>/dev/null || true; fi
}}
trap cleanup EXIT INT TERM HUP

{shell_join(launched)} &
bench_pid=$!

monitor() {{
    while kill -0 "$bench_pid" 2>/dev/null; do
        epoch_ms=$(date +%s%3N 2>/dev/null || date +%s000)
        proc_mib=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory \\
            --format=csv,noheader,nounits 2>/dev/null | \\
            awk -F',' -v wanted="$bench_pid" '$1 + 0 == wanted {{ gsub(/[[:space:]]/, "", $2); print $2; exit }}')
        gpu_csv=$(nvidia-smi -i {shlex.quote(str(gpu_id))} \\
            --query-gpu=memory.used,clocks.current.graphics,temperature.gpu,power.draw,utilization.gpu \\
            --format=csv,noheader,nounits 2>/dev/null | head -n 1)
        IFS=',' read -r device_mib clock_mhz temp_c power_w util_pct <<< "$gpu_csv"
        trim() {{ local value="$1"; value="${{value//[[:space:]]/}}"; printf '%s' "$value"; }}
        printf '{TELEMETRY_PREFIX}%s|%s|%s|%s|%s|%s|%s|%s\\n' \\
            "$(trim "$epoch_ms")" "$bench_pid" "$(trim "$proc_mib")" \\
            "$(trim "$device_mib")" "$(trim "$clock_mhz")" "$(trim "$temp_c")" \\
            "$(trim "$power_w")" "$(trim "$util_pct")" >&2
{system_probe}        sleep {interval:.6f}
    done
}}
monitor &
monitor_pid=$!
{pcie_dmon_start}

wait "$bench_pid"
bench_rc=$?
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=''
{pcie_dmon_cleanup}
bench_pid=''
printf '{EXIT_PREFIX}%s\\n' "$bench_rc" >&2
exit "$bench_rc"
"""


def parse_optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "[N/A]", "NAN"}:
        return None
    try:
        result = float(stripped)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def parse_telemetry_line(line: str) -> TelemetrySample | None:
    if line.startswith(PCIE_TELEMETRY_PREFIX):
        fields = line.rstrip("\n").split("|")[1:]
        if len(fields) != 3:
            return None
        try:
            epoch_ms = int(fields[0])
        except ValueError:
            epoch_ms = None
        return TelemetrySample(
            epoch_ms=epoch_ms,
            pid=None,
            process_mib=None,
            device_mib=None,
            clock_mhz=None,
            temperature_c=None,
            power_w=None,
            utilization_pct=None,
            # nvidia-smi dmon documents rxpci/txpci in MB/s.  Preserve the
            # numeric value as MiB/s-compatible diagnostic telemetry; backend
            # CUDA byte/event timing remains the authoritative throughput.
            pcie_rx_mib_s=parse_optional_float(fields[1]),
            pcie_tx_mib_s=parse_optional_float(fields[2]),
        )
    if line.startswith(SYSTEM_TELEMETRY_PREFIX):
        values = {
            key: value
            for item in line.rstrip("\n").split("|")[1:]
            if "=" in item
            for key, value in [item.split("=", 1)]
        }

        def optional_int(value: str) -> int | None:
            try:
                return int(value.strip())
            except (AttributeError, ValueError):
                return None

        rx_kib_s = parse_optional_float(values.get("pcie_rx_kib_s", ""))
        tx_kib_s = parse_optional_float(values.get("pcie_tx_kib_s", ""))
        return TelemetrySample(
            epoch_ms=optional_int(values.get("epoch_ms", "")),
            pid=optional_int(values.get("pid", "")),
            process_mib=None,
            device_mib=None,
            clock_mhz=None,
            temperature_c=None,
            power_w=None,
            utilization_pct=None,
            memory_utilization_pct=parse_optional_float(values.get("memory_util_pct", "")),
            memory_clock_mhz=parse_optional_float(values.get("memory_clock_mhz", "")),
            pcie_rx_mib_s=rx_kib_s / 1024.0 if rx_kib_s is not None else None,
            pcie_tx_mib_s=tx_kib_s / 1024.0 if tx_kib_s is not None else None,
            pcie_link_gen=parse_optional_float(values.get("pcie_link_gen", "")),
            pcie_link_width=parse_optional_float(values.get("pcie_link_width", "")),
            pstate=values.get("pstate", "").strip() or None,
            process_rss_mib=parse_optional_float(values.get("proc_rss_mib", "")),
            process_locked_mib=parse_optional_float(values.get("proc_locked_mib", "")),
            process_cpu_pct=None,
            process_cpu_ticks=parse_optional_float(values.get("proc_cpu_ticks", "")),
            total_cpu_ticks=parse_optional_float(values.get("total_cpu_ticks", "")),
            cpu_count=parse_optional_float(values.get("cpu_count", "")),
            process_read_mib=parse_optional_float(values.get("proc_read_mib", "")),
            process_write_mib=parse_optional_float(values.get("proc_write_mib", "")),
            process_major_faults=parse_optional_float(values.get("proc_major_faults", "")),
            host_mem_available_mib=parse_optional_float(values.get("mem_available_mib", "")),
            host_swap_free_mib=parse_optional_float(values.get("swap_free_mib", "")),
            psi_cpu_some_pct=parse_optional_float(values.get("psi_cpu_some_pct", "")),
            psi_memory_some_pct=parse_optional_float(values.get("psi_memory_some_pct", "")),
            psi_memory_full_pct=parse_optional_float(values.get("psi_memory_full_pct", "")),
            psi_io_some_pct=parse_optional_float(values.get("psi_io_some_pct", "")),
            psi_io_full_pct=parse_optional_float(values.get("psi_io_full_pct", "")),
        )
    if not line.startswith(TELEMETRY_PREFIX):
        return None
    fields = line.rstrip("\n").split("|")[1:]
    if len(fields) != 8:
        return None

    def optional_int(value: str) -> int | None:
        try:
            return int(value.strip())
        except ValueError:
            return None

    return TelemetrySample(
        epoch_ms=optional_int(fields[0]),
        pid=optional_int(fields[1]),
        process_mib=parse_optional_float(fields[2]),
        device_mib=parse_optional_float(fields[3]),
        clock_mhz=parse_optional_float(fields[4]),
        temperature_c=parse_optional_float(fields[5]),
        power_w=parse_optional_float(fields[6]),
        utilization_pct=parse_optional_float(fields[7]),
    )


def parse_backend_profile_line(line: str) -> BackendProfileEvent | None:
    stripped = line.strip()
    if not stripped.startswith(BACKEND_PROFILE_PREFIX):
        return None
    values = parse_key_values(stripped)
    event = values.get("event", "")
    if not event:
        return None
    return BackendProfileEvent(event=event, values=values, raw=stripped)


def merge_telemetry_samples(samples: Sequence[TelemetrySample]) -> list[TelemetrySample]:
    """Merge the GPU and host records emitted for the same polling instant."""

    grouped: dict[tuple[int | None, int | None], TelemetrySample] = {}
    for sample in samples:
        key = sample.epoch_ms, sample.pid
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = sample
            continue
        combined: dict[str, Any] = {}
        for name in TelemetrySample.__dataclass_fields__:
            newer = getattr(sample, name)
            combined[name] = newer if newer is not None else getattr(previous, name)
        grouped[key] = TelemetrySample(**combined)
    return [grouped[key] for key in sorted(grouped, key=lambda value: ((value[0] or 0), (value[1] or 0)))]


def summarize_telemetry(samples: Sequence[TelemetrySample]) -> dict[str, str]:
    result: dict[str, str] = {"telemetry_samples": str(len(samples))}

    def values(attribute: str) -> list[float]:
        return [
            float(value)
            for sample in samples
            if (value := getattr(sample, attribute)) is not None
        ]

    def set_peak(field_name: str, attribute: str) -> None:
        entries = values(attribute)
        result[field_name] = f"{max(entries):.3f}" if entries else ""

    def set_mean_min_max(prefix: str, attribute: str, include_min: bool = True) -> None:
        entries = values(attribute)
        result[f"{prefix}_mean"] = f"{sum(entries) / len(entries):.3f}" if entries else ""
        if include_min:
            result[f"{prefix}_min"] = f"{min(entries):.3f}" if entries else ""
        result[f"{prefix}_max"] = f"{max(entries):.3f}" if entries else ""

    set_peak("gpu_process_peak_mib", "process_mib")
    set_peak("gpu_device_peak_mib", "device_mib")
    set_mean_min_max("graphics_clock_mhz", "clock_mhz")
    set_mean_min_max("temperature_c", "temperature_c", include_min=False)
    set_mean_min_max("power_w", "power_w", include_min=False)
    set_mean_min_max("gpu_util_pct", "utilization_pct", include_min=False)

    # Map generic names to the stable CSV schema.
    return {
        "telemetry_samples": result.get("telemetry_samples", "0"),
        "gpu_process_peak_mib": result.get("gpu_process_peak_mib", ""),
        "gpu_device_peak_mib": result.get("gpu_device_peak_mib", ""),
        "graphics_clock_mean_mhz": result.get("graphics_clock_mhz_mean", ""),
        "graphics_clock_min_mhz": result.get("graphics_clock_mhz_min", ""),
        "graphics_clock_max_mhz": result.get("graphics_clock_mhz_max", ""),
        "temperature_mean_c": result.get("temperature_c_mean", ""),
        "temperature_max_c": result.get("temperature_c_max", ""),
        "power_mean_w": result.get("power_w_mean", ""),
        "power_max_w": result.get("power_w_max", ""),
        "gpu_util_mean_pct": result.get("gpu_util_pct_mean", ""),
        "gpu_util_max_pct": result.get("gpu_util_pct_max", ""),
    }


def _finite_values(samples: Sequence[TelemetrySample], attribute: str) -> list[float]:
    return [
        float(value)
        for sample in samples
        if (value := getattr(sample, attribute)) is not None and math.isfinite(float(value))
    ]


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def telemetry_profile_summary(samples: Sequence[TelemetrySample]) -> dict[str, Any]:
    """Return JSON-native system diagnostics for a profile sidecar record."""

    ordered = sorted(
        (sample for sample in samples if sample.epoch_ms is not None),
        key=lambda sample: int(sample.epoch_ms or 0),
    )
    cpu_pct: list[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        if previous.pid != current.pid:
            continue
        required = (
            previous.process_cpu_ticks,
            current.process_cpu_ticks,
            previous.total_cpu_ticks,
            current.total_cpu_ticks,
            current.cpu_count,
        )
        if any(value is None for value in required):
            continue
        process_delta = float(current.process_cpu_ticks) - float(previous.process_cpu_ticks)
        total_delta = float(current.total_cpu_ticks) - float(previous.total_cpu_ticks)
        if process_delta >= 0 and total_delta > 0 and float(current.cpu_count) > 0:
            cpu_pct.append(100.0 * process_delta * float(current.cpu_count) / total_delta)

    def delta(attribute: str) -> float | None:
        values = _finite_values(ordered, attribute)
        return max(0.0, values[-1] - values[0]) if len(values) >= 2 else None

    pstate_counts: dict[str, int] = {}
    for sample in samples:
        if sample.pstate:
            pstate_counts[sample.pstate] = pstate_counts.get(sample.pstate, 0) + 1
    epochs = [int(sample.epoch_ms) for sample in samples if sample.epoch_ms is not None]
    return {
        "sample_count": len(samples),
        "first_epoch_ms": min(epochs) if epochs else None,
        "last_epoch_ms": max(epochs) if epochs else None,
        "sample_span_ms": max(epochs) - min(epochs) if len(epochs) >= 2 else None,
        "gpu": {
            "process_vram_mib": _stats(_finite_values(samples, "process_mib")),
            "device_vram_mib": _stats(_finite_values(samples, "device_mib")),
            "graphics_clock_mhz": _stats(_finite_values(samples, "clock_mhz")),
            "memory_clock_mhz": _stats(_finite_values(samples, "memory_clock_mhz")),
            "temperature_c": _stats(_finite_values(samples, "temperature_c")),
            "power_w": _stats(_finite_values(samples, "power_w")),
            "compute_util_pct": _stats(_finite_values(samples, "utilization_pct")),
            "memory_util_pct": _stats(_finite_values(samples, "memory_utilization_pct")),
            "pstate_counts": pstate_counts,
        },
        "pcie": {
            "rx_mib_s": _stats(_finite_values(samples, "pcie_rx_mib_s")),
            "tx_mib_s": _stats(_finite_values(samples, "pcie_tx_mib_s")),
            "link_gen": _stats(_finite_values(samples, "pcie_link_gen")),
            "link_width": _stats(_finite_values(samples, "pcie_link_width")),
            "counter_note": (
                "nvidia-smi pcie.rx_util/tx_util converted from KiB/s; null when the "
                "driver does not expose these optional fields"
            ),
        },
        "process": {
            "rss_mib": _stats(_finite_values(samples, "process_rss_mib")),
            "locked_mib": _stats(_finite_values(samples, "process_locked_mib")),
            "cpu_pct": _stats(cpu_pct),
            "read_mib_delta": delta("process_read_mib"),
            "write_mib_delta": delta("process_write_mib"),
            "major_faults_delta": delta("process_major_faults"),
        },
        "host": {
            "mem_available_mib": _stats(_finite_values(samples, "host_mem_available_mib")),
            "swap_free_mib": _stats(_finite_values(samples, "host_swap_free_mib")),
            "psi_avg10_pct": {
                "cpu_some": _stats(_finite_values(samples, "psi_cpu_some_pct")),
                "memory_some": _stats(_finite_values(samples, "psi_memory_some_pct")),
                "memory_full": _stats(_finite_values(samples, "psi_memory_full_pct")),
                "io_some": _stats(_finite_values(samples, "psi_io_some_pct")),
                "io_full": _stats(_finite_values(samples, "psi_io_full_pct")),
            },
        },
    }


def backend_profile_summary(events: Sequence[BackendProfileEvent]) -> dict[str, Any]:
    """Aggregate the version-1 ``hybrid_profile`` protocol.

    Current backends emit ``dma``/``wait`` detail records, periodic and final
    ``summary`` records, and final per-``layer`` records.  Early prototypes
    emitted a single ``sample`` record instead.  Supporting all four forms is
    intentional: a sidecar remains readable while the profiler is deployed to
    a machine with a slightly older temporary build.

    A final ``scope=total`` summary overlaps every window summary and every
    layer record.  It is therefore authoritative and must never be added to
    them.  If it is absent (for example, an interrupted process), complete
    windows are summed; detail records are the last fallback.
    """

    counts: dict[str, int] = {}
    for event in events:
        counts[event.event] = counts.get(event.event, 0) + 1

    def number(values: dict[str, str], *names: str) -> float | None:
        for name in names:
            parsed = parse_optional_float(values.get(name, ""))
            if parsed is not None:
                return parsed
        return None

    configs = [event.values for event in events if event.event == "config"]
    summaries_all = [event for event in events if event.event == "summary"]
    total_summaries = [event for event in summaries_all if event.values.get("scope") == "total"]
    window_summaries = [event for event in summaries_all if event.values.get("scope") != "total"]
    if total_summaries:
        summaries = total_summaries
        aggregation_source = "summary_total"
    elif window_summaries:
        summaries = window_summaries
        aggregation_source = "summary_windows"
    else:
        summaries = []
        aggregation_source = "detail_events"

    detail_dma = [event for event in events if event.event == "dma"]
    detail_wait = [event for event in events if event.event == "wait"]
    detail_zero = [event for event in events if event.event == "zero_copy"]
    legacy_samples = [event for event in events if event.event == "sample"]
    for event in legacy_samples:
        mode = event.values.get("mode", "")
        kind = event.values.get("kind", event.values.get("sample_kind", ""))
        is_zero = mode == "zero_copy" or kind in {"zero_copy", "zc"}
        if is_zero:
            detail_zero.append(event)
        else:
            detail_dma.append(event)
            if number(event.values, "wait_ms") is not None:
                detail_wait.append(event)

    copy_ms = [
        value
        for event in detail_dma
        if (value := number(event.values, "copy_ms", "dma_ms")) is not None
    ]
    ready_ms = [
        value
        for event in detail_dma
        if (value := number(event.values, "kick_to_ready_ms", "ready_ms")) is not None
    ]
    wait_ms = [
        value
        for event in detail_wait
        if (value := number(event.values, "wait_ms")) is not None
    ]

    def sum_summary(*names: str) -> float | None:
        values = [
            value
            for event in summaries
            if (value := number(event.values, *names)) is not None
        ]
        return sum(values) if values else None

    def weighted_summary(value_names: tuple[str, ...], weight_names: tuple[str, ...]) -> float | None:
        weighted: list[tuple[float, float]] = []
        for event in summaries:
            value = number(event.values, *value_names)
            weight = number(event.values, *weight_names)
            if value is not None:
                weighted.append((value, weight if weight is not None and weight > 0 else 1.0))
        if not weighted:
            return None
        return sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted)

    # copy_count/total_bytes are actual operation totals.  sampled_* and all
    # CUDA-event durations cover only sampled operations when sample_every>1.
    dma_count = sum_summary("copy_count", "dma_count")
    sampled_dma_count = sum_summary("sampled_copy_count")
    dma_bytes = sum_summary("total_bytes", "dma_bytes")
    sampled_dma_bytes = sum_summary("sampled_bytes")
    dma_ms = sum_summary("copy_ms", "dma_ms")
    if dma_count is None:
        dma_count = float(len(detail_dma))
    if sampled_dma_count is None:
        sampled_dma_count = float(len(detail_dma))
    if dma_bytes is None:
        byte_values = [number(event.values, "bytes", "total_bytes") for event in detail_dma]
        dma_bytes = sum(value for value in byte_values if value is not None)
    if sampled_dma_bytes is None:
        sampled_dma_bytes = sum(
            number(event.values, "bytes", "sampled_bytes") or 0 for event in detail_dma
        )
    if dma_ms is None:
        dma_ms = sum(copy_ms)
    effective_gbps = weighted_summary(("effective_gbps",), ("copy_ms", "sampled_bytes"))
    if effective_gbps is None and sampled_dma_bytes is not None and dma_ms is not None and dma_ms > 0:
        effective_gbps = sampled_dma_bytes / dma_ms / 1_000_000.0

    wait_count = sum_summary("wait_count")
    wait_total_ms = sum_summary("wait_ms")
    if wait_count is None:
        wait_count = float(len(detail_wait))
    if wait_total_ms is None:
        wait_total_ms = sum(wait_ms)
    hits = sum_summary("prefetch_hits", "hits")
    misses = sum_summary("prefetch_misses", "misses")
    if hits is None:
        hit_values = [number(event.values, "prefetch_hit") for event in detail_wait]
        hits = sum(1 for value in hit_values if value is not None and value != 0)
        misses = sum(1 for value in hit_values if value is not None and value == 0)

    hidden_pct = weighted_summary(("overlap_pct", "hidden_pct"), ("ready_ms", "copy_ms"))
    if hidden_pct is None:
        detailed_hidden = [
            (value, number(event.values, "kick_to_ready_ms", "ready_ms") or 1.0)
            for event in detail_dma
            if (value := number(event.values, "overlap_pct", "hidden_pct")) is not None
        ]
        if detailed_hidden:
            hidden_pct = (
                sum(value * weight for value, weight in detailed_hidden)
                / sum(weight for _, weight in detailed_hidden)
            )
        elif dma_ms and wait_total_ms is not None:
            hidden_pct = max(0.0, 100.0 * (1.0 - wait_total_ms / dma_ms))

    zero_copy_touches = sum_summary("zero_copy_touches", "zc_touches")
    zero_copy_bytes = sum_summary("zero_copy_bytes", "zc_bytes", "estimated_read_bytes")
    sampled_zero_copy = sum_summary("sampled_zero_copy")
    sampled_zero_copy_bytes = sum_summary("sampled_zero_copy_bytes")
    if zero_copy_touches is None:
        zero_copy_touches = float(len(detail_zero))
    if zero_copy_bytes is None:
        zero_copy_bytes = sum(
            number(event.values, "remote_bytes", "estimated_read_bytes", "bytes") or 0
            for event in detail_zero
        )
    if sampled_zero_copy is None:
        sampled_zero_copy = float(len(detail_zero))
    if sampled_zero_copy_bytes is None:
        sampled_zero_copy_bytes = sum(
            number(event.values, "remote_bytes", "estimated_read_bytes", "bytes") or 0
            for event in detail_zero
        )
    dropped = sum_summary("dropped") or 0.0

    layer_records: list[dict[str, Any]] = []
    for event in events:
        if event.event != "layer":
            continue
        values = event.values
        layer_value = number(values, "layer")
        layer_records.append(
            {
                "layer": int(layer_value) if layer_value is not None else None,
                "parity": int(number(values, "parity") or 0),
                "copy_count": int(number(values, "copy_count", "dma_count") or 0),
                "total_bytes": int(number(values, "total_bytes", "dma_bytes") or 0),
                "sampled_copy_count": int(number(values, "sampled_copy_count") or 0),
                "sampled_bytes": int(number(values, "sampled_bytes") or 0),
                "copy_ms": number(values, "copy_ms", "dma_ms"),
                "ready_ms": number(values, "ready_ms"),
                "queue_ms": number(values, "queue_ms"),
                "wait_count": int(number(values, "wait_count") or 0),
                "wait_ms": number(values, "wait_ms"),
                "work_ms": number(values, "work_ms", "work_window_ms"),
                "effective_gbps_decimal": number(values, "effective_gbps"),
                "overlap_pct": number(values, "overlap_pct", "hidden_pct"),
                "stall_pct": number(values, "stall_pct"),
                "prefetch_hits": int(number(values, "prefetch_hits") or 0),
                "prefetch_misses": int(number(values, "prefetch_misses") or 0),
                "zero_copy_touches": int(number(values, "zero_copy_touches", "zc_touches") or 0),
                "zero_copy_bytes": int(number(values, "zero_copy_bytes", "zc_bytes") or 0),
            }
        )
    slowest_layers = sorted(
        layer_records,
        key=lambda record: float(record["wait_ms"] or 0.0),
        reverse=True,
    )[:8]

    # Per-evaluation records are intentionally not folded into totals: like
    # layer records they are an alternate grouping of the same events.  They
    # expose which prompt/decode evaluation paid the PCIe stall and also make
    # cross-eval wrap-prefetch attribution visible via detail work_eval IDs.
    eval_records: list[dict[str, Any]] = []
    for event in events:
        if event.event != "eval":
            continue
        values = event.values
        eval_value = number(values, "eval")
        eval_records.append(
            {
                "eval": int(eval_value) if eval_value is not None else None,
                "copy_count": int(number(values, "copy_count", "dma_count") or 0),
                "total_bytes": int(number(values, "total_bytes", "dma_bytes") or 0),
                "sampled_copy_count": int(number(values, "sampled_copy_count") or 0),
                "sampled_bytes": int(number(values, "sampled_bytes") or 0),
                "copy_ms": number(values, "copy_ms", "dma_ms"),
                "ready_ms": number(values, "ready_ms"),
                "queue_ms": number(values, "queue_ms"),
                "wait_count": int(number(values, "wait_count") or 0),
                "wait_ms": number(values, "wait_ms"),
                "work_ms": number(values, "work_ms", "work_window_ms"),
                "effective_gbps_decimal": number(values, "effective_gbps"),
                "overlap_pct": number(values, "overlap_pct", "hidden_pct"),
                "stall_pct": number(values, "stall_pct"),
                "prefetch_hits": int(number(values, "prefetch_hits") or 0),
                "prefetch_misses": int(number(values, "prefetch_misses") or 0),
                "zero_copy_touches": int(number(values, "zero_copy_touches", "zc_touches") or 0),
                "zero_copy_bytes": int(number(values, "zero_copy_bytes", "zc_bytes") or 0),
                "dropped": int(number(values, "dropped") or 0),
            }
        )
    eval_records.sort(key=lambda record: int(record["eval"] or 0))

    cross_eval_samples = 0
    for event in events:
        if event.event not in {"dma", "wait", "zero_copy", "sample"}:
            continue
        eval_id = number(event.values, "eval")
        work_eval_id = number(event.values, "work_eval")
        if eval_id is not None and work_eval_id is not None and work_eval_id != 0 and eval_id != work_eval_id:
            cross_eval_samples += 1

    return {
        "event_count": len(events),
        "event_counts": counts,
        "protocol_versions": sorted(
            {event.values["version"] for event in events if event.values.get("version")}
        ),
        "aggregation_source": aggregation_source,
        "summary_records_seen": len(summaries_all),
        "summary_records_used": len(summaries),
        "config_records": configs,
        "dropped": int(dropped),
        "dma": {
            "count": int(dma_count),
            "bytes": int(dma_bytes or 0),
            "sampled_count": int(sampled_dma_count),
            "sampled_bytes": int(sampled_dma_bytes or 0),
            "copy_total_ms": dma_ms,
            "timing_is_sampled": bool(
                int(sampled_dma_count) != int(dma_count)
                or int(sampled_dma_bytes or 0) != int(dma_bytes or 0)
            ),
            "effective_gbps_decimal": effective_gbps,
            "copy_p50_ms": _percentile(copy_ms, 0.50),
            "copy_p95_ms": _percentile(copy_ms, 0.95),
            "copy_max_ms": max(copy_ms) if copy_ms else None,
            "kick_to_ready_p95_ms": _percentile(ready_ms, 0.95),
            "wait_count": int(wait_count),
            "wait_total_ms": wait_total_ms,
            "wait_p95_ms": _percentile(wait_ms, 0.95),
            "wait_max_ms": max(wait_ms) if wait_ms else None,
            "hidden_pct": hidden_pct,
            "overlap_pct": hidden_pct,
            "prefetch_hits": int(hits or 0),
            "prefetch_misses": int(misses or 0),
            "prefetch_hit_pct": (
                100.0 * hits / (hits + misses)
                if hits is not None and misses is not None and hits + misses > 0
                else None
            ),
        },
        "zero_copy": {
            "touches": int(zero_copy_touches or 0),
            "bytes": int(zero_copy_bytes or 0),
            "sampled_touches": int(sampled_zero_copy or 0),
            "sampled_bytes": int(sampled_zero_copy_bytes or 0),
        },
        "layers": layer_records,
        "slowest_layers_by_wait_ms": slowest_layers,
        "evals": eval_records,
        "cross_eval_detail_samples": cross_eval_samples,
    }


def profile_sidecar_record(
    *,
    profile: dict[str, Any],
    run_id: str,
    manifest_sha256: str,
    case_key: dict[str, Any],
    status: str,
    result: CommandResult,
    command_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    telemetry = telemetry_profile_summary(result.telemetry)
    backend = backend_profile_summary(result.backend_profile)
    gpu = telemetry["gpu"]
    pcie = telemetry["pcie"]
    process = telemetry["process"]
    return {
        "schema_version": PROFILE_SIDECAR_SCHEMA_VERSION,
        "protocol": "intertidal-profile-v1",
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "command_sha256": hashlib.sha256(
            json.dumps(command_identity or {}, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "case_key": case_key,
        "status": status,
        "profile_mode": profile["mode"],
        "profile_config": {
            key: profile[key]
            for key in (
                "interval_s", "force_direct", "pcie_dmon", "raw_sidecar", "backend_enabled",
                "backend_sample_every", "backend_summary_every", "backend_ring", "backend_detail",
            )
        },
        "started_epoch_ms": result.started_epoch_ms,
        "ended_epoch_ms": result.ended_epoch_ms,
        "elapsed_s": result.elapsed_s,
        "telemetry": telemetry,
        "backend": backend,
        "diagnosis_inputs": {
            "backend_dma_effective_gbps": backend["dma"]["effective_gbps_decimal"],
            "backend_dma_hidden_pct": backend["dma"]["hidden_pct"],
            "backend_dma_wait_total_ms": backend["dma"]["wait_total_ms"],
            "pcie_rx_max_mib_s": pcie["rx_mib_s"]["max"],
            "pcie_tx_max_mib_s": pcie["tx_mib_s"]["max"],
            "gpu_compute_util_mean_pct": gpu["compute_util_pct"]["mean"],
            "gpu_memory_util_mean_pct": gpu["memory_util_pct"]["mean"],
            "gpu_graphics_clock_min_mhz": gpu["graphics_clock_mhz"]["min"],
            "process_cpu_max_pct": process["cpu_pct"]["max"],
            "process_major_faults_delta": process["major_faults_delta"],
        },
        "raw": {
            "telemetry_samples": [asdict(sample) for sample in result.telemetry],
            "backend_records": [event.raw for event in result.backend_profile],
        }
        if profile.get("raw_sidecar", False)
        else None,
    }


def append_profile_sidecar(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _stream_pipe(
    pipe: TextIO,
    sink: TextIO,
    chunks: list[str],
    telemetry_queue: queue.SimpleQueue[TelemetrySample],
    show_telemetry: bool,
) -> None:
    try:
        for line in iter(pipe.readline, ""):
            chunks.append(line)
            sample = parse_telemetry_line(line)
            if sample is not None:
                telemetry_queue.put(sample)
                if not show_telemetry:
                    continue
            if line.startswith(BACKEND_PROFILE_PREFIX) and not show_telemetry:
                continue
            if line.startswith(EXIT_PREFIX) and not show_telemetry:
                continue
            sink.write(line)
            sink.flush()
    finally:
        pipe.close()


def execute_wrapper(
    manifest: dict[str, Any],
    wrapper: str,
    *,
    timeout_s: float,
    show_telemetry: bool,
) -> CommandResult:
    remote = manifest["remote"]
    if remote.get("transport", "ssh") == "local":
        launcher = ["bash", "-s"]
    else:
        launcher = ["ssh", "-T"]
        launcher.extend(str(item) for item in remote.get("ssh_options", []))
        launcher.extend([str(remote["target"]), "bash", "-s"])

    started = time.monotonic()
    started_epoch_ms = time.time_ns() // 1_000_000
    proc = subprocess.Popen(
        launcher,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    proc.stdin.write(wrapper)
    proc.stdin.close()

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    samples: queue.SimpleQueue[TelemetrySample] = queue.SimpleQueue()
    threads = [
        threading.Thread(
            target=_stream_pipe,
            args=(proc.stdout, sys.stdout, stdout_chunks, samples, show_telemetry),
            daemon=True,
        ),
        threading.Thread(
            target=_stream_pipe,
            args=(proc.stderr, sys.stderr, stderr_chunks, samples, show_telemetry),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        returncode = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(f"case timed out after {timeout_s:.1f}s; terminating remote wrapper", file=sys.stderr)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            returncode = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            returncode = proc.wait()
    for thread in threads:
        thread.join(timeout=5)

    telemetry: list[TelemetrySample] = []
    while not samples.empty():
        telemetry.append(samples.get())
    telemetry = merge_telemetry_samples(telemetry)
    stderr = "".join(stderr_chunks)
    return CommandResult(
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr=stderr,
        elapsed_s=time.monotonic() - started,
        timed_out=timed_out,
        telemetry=telemetry,
        backend_profile=[
            event
            for line in stderr.splitlines()
            if (event := parse_backend_profile_line(line)) is not None
        ],
        started_epoch_ms=started_epoch_ms,
        ended_epoch_ms=time.time_ns() // 1_000_000,
    )


def parse_bench(stdout: str) -> dict[str, float]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("build_commit,")
    )
    header = next(csv.reader([lines[header_index]]))
    records: list[dict[str, str]] = []
    for line in lines[header_index + 1 :]:
        try:
            values = next(csv.reader([line]))
        except csv.Error:
            continue
        if len(values) != len(header):
            continue
        record = dict(zip(header, values))
        try:
            int(record["n_prompt"])
            int(record["n_gen"])
            float(record["avg_ts"])
        except (KeyError, ValueError):
            continue
        records.append(record)

    prefill = next(row for row in records if int(row["n_prompt"]) > 0)
    decode = next(row for row in records if int(row["n_gen"]) > 0)
    parsed = {
        "prefill_tok_s": float(prefill["avg_ts"]),
        "prefill_std_tok_s": float(prefill["stddev_ts"]),
        "decode_tok_s": float(decode["avg_ts"]),
        "decode_std_tok_s": float(decode["stddev_ts"]),
    }
    for key in ("prefill_tok_s", "decode_tok_s"):
        if not math.isfinite(parsed[key]) or parsed[key] <= 0:
            raise ArithmeticError(f"non-finite or non-positive benchmark result: {key}={parsed[key]}")
    return parsed


def parse_key_values(line: str) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for item in line.strip().split(",")[1:]
        if "=" in item
        for key, value in [item.split("=", 1)]
    }


def parse_hybrid_layout(stderr: str) -> dict[str, Any]:
    layouts = [
        parse_key_values(line)
        for line in stderr.splitlines()
        if line.startswith("hybrid_vmm,")
    ]
    if not layouts:
        return {}

    sums = {
        "requested_pages": "requested_pages",
        "gross_pages": "gross_pages",
        "gross_bytes": "gross_remote_bytes",
        "staging_bytes": "staging_bytes",
        "net_bytes": "net_saved_bytes",
        "selected_tensors": "selected_tensors",
        "layers": "selected_layers",
    }
    result: dict[str, Any] = {"buffers": layouts}
    for source, destination in sums.items():
        values = [int(layout[source]) for layout in layouts if source in layout]
        if values:
            result[destination] = sum(values)
    result["mode"] = layouts[-1].get("mode", "")
    return result


def parse_native_buffers(stderr: str) -> dict[str, float]:
    result: dict[str, float] = {}
    pattern = re.compile(r"^.*?\s+(CPU(?:_Mapped)?|CUDA\d+) model buffer size =\s+([0-9.]+) MiB")
    for line in stderr.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        backend, value = match.groups()
        if backend.startswith("CPU"):
            result["native_cpu_model_mib"] = result.get("native_cpu_model_mib", 0.0) + float(value)
        else:
            result["native_cuda_model_mib"] = result.get("native_cuda_model_mib", 0.0) + float(value)
    return result


def classify_result(result: CommandResult) -> tuple[str, dict[str, float], str]:
    combined = result.stdout + "\n" + result.stderr
    if result.timed_out:
        return "timeout", {}, "case timeout"
    # OpenSSH reserves exit status 255 for transport/client failures.  A dead
    # control master can produce no diagnostic at all, so classify the status
    # itself before attempting to parse benchmark output.
    if result.returncode == 255:
        return "transport_error", {}, last_error_lines(combined) or "SSH exited with status 255"
    if TRANSPORT_ERROR_RE.search(combined):
        return "transport_error", {}, last_error_lines(combined)
    if UNSUPPORTED_RE.search(combined):
        return "unsupported", {}, last_error_lines(combined)
    if OOM_RE.search(combined):
        return "oom", {}, last_error_lines(combined)
    try:
        parsed = parse_bench(result.stdout)
    except ArithmeticError as exc:
        return "correctness_fail", {}, str(exc)
    except (StopIteration, KeyError, ValueError, csv.Error) as exc:
        status = "error" if result.returncode else "parse_error"
        return status, {}, f"{type(exc).__name__}: {exc}; {last_error_lines(combined)}"
    if result.returncode:
        return "error", parsed, last_error_lines(combined)
    return "ok", parsed, ""


def last_error_lines(text: str, limit: int = 1000) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.startswith(TELEMETRY_PREFIX)
        and not line.startswith(SYSTEM_TELEMETRY_PREFIX)
        and not line.startswith(PCIE_TELEMETRY_PREFIX)
        and not line.startswith(EXIT_PREFIX)
        and not line.startswith(BACKEND_PROFILE_PREFIX)
    ]
    return " | ".join(lines[-8:])[-limit:]


def blank_row() -> dict[str, str]:
    return {field: "" for field in CSV_FIELDS}


def baseline_from_rows(
    rows: dict[tuple[str, str, str], dict[str, str]], scheme: str
) -> tuple[float, float, str] | None:
    candidates: list[tuple[Decimal, dict[str, str]]] = []
    for (row_scheme, workset, control), row in rows.items():
        if row_scheme != scheme or row.get("status") != "ok":
            continue
        is_initial = control in {"pp:0", "ngl:61"}
        if is_initial:
            candidates.append((Decimal(workset), row))
    if not candidates:
        return None
    workset, row = min(candidates, key=lambda item: item[0])
    return float(row["prefill_tok_s"]), float(row["decode_tok_s"]), canonical_decimal(workset)


def native_all_local_peak_from_rows(
    manifest: dict[str, Any],
    rows: dict[tuple[str, str, str], dict[str, str]],
    context_tokens: int,
) -> float | None:
    """Return a measured, same-context native all-local process peak.

    A workset label is deliberately not used as the join key: context rounding
    can make two requested labels select the same allocation.  When repeated
    controls exist, the largest sampled peak is the conservative reference.
    """

    start = int(
        manifest["sweep"].get(
            "native_ngl_start", manifest["model"].get("gpu_layers", 61)
        )
    )
    candidates: list[float] = []
    for row in rows.values():
        if (
            row.get("scheme") != "native_cpu_layers"
            or row.get("status") != "ok"
            or row.get("native_ngl") != str(start)
            or row.get("context_tokens") != str(context_tokens)
        ):
            continue
        try:
            peak = float(row["gpu_process_peak_mib"])
        except (KeyError, ValueError):
            continue
        if math.isfinite(peak) and peak > 0:
            candidates.append(peak)
    return max(candidates) if candidates else None


def row_for_result(
    manifest: dict[str, Any],
    manifest_hash: str,
    run_id: str,
    case: Case,
    env: dict[str, str],
    command: list[str],
    result: CommandResult,
    baseline: tuple[float, float, str] | None,
    correctness: CorrectnessEvaluation | None = None,
    native_reference_peak_mib: float | None = None,
) -> dict[str, str]:
    status, bench, error = classify_result(result)
    if correctness is not None and correctness.status == "fail" and status == "ok":
        status = "correctness_fail"
        error = correctness.error
    layout = parse_hybrid_layout(result.stderr)
    native = parse_native_buffers(result.stderr)
    model_bytes = int(manifest["model"]["tensor_bytes"])

    row = blank_row()
    row.update(
        schema_version=str(SCHEMA_VERSION),
        run_id=run_id,
        case_time_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=manifest_hash,
        scheme=case.scheme,
        phase=case.phase,
        workset_target_gib=f"{case.workset_gib:.1f}",
        calibrated_all_local_mib=f"{calibrated_peak_mib(manifest, case.context_tokens):.3f}",
        context_tokens=str(case.context_tokens),
        target_offload_pp=(
            f"{case.target_offload_pp:.1f}" if case.target_offload_pp is not None else ""
        ),
        page_budget=str(case.page_budget) if case.page_budget is not None else "",
        native_ngl=str(case.native_ngl) if case.native_ngl is not None else "",
        status=status,
        exit_code=str(result.returncode),
        elapsed_s=f"{result.elapsed_s:.3f}",
        command_json=json.dumps(
            {"env": env, "argv": command}, separators=(",", ":"), sort_keys=True
        ),
        layout_json=json.dumps(layout, separators=(",", ":"), sort_keys=True) if layout else "",
        error_tail=error,
    )
    if correctness is not None:
        row["checksum_status"] = correctness.status
        row["checksum_baseline_sha256"] = correctness.baseline_sha256
        if correctness.nonfinite_logits is not None:
            row["nonfinite_logits"] = str(correctness.nonfinite_logits)
        if correctness.entries:
            row["checksum_sequence_json"] = json.dumps(
                [
                    {
                        "label": entry.label,
                        "checksum": entry.checksum,
                        "nonfinite": entry.nonfinite,
                    }
                    for entry in correctness.entries
                ],
                separators=(",", ":"),
            )
    row.update(summarize_telemetry(result.telemetry))

    for key, value in bench.items():
        row[key] = f"{value:.6f}"
    for key, value in native.items():
        row[key] = f"{value:.3f}"

    for key in (
        "requested_pages",
        "gross_pages",
        "gross_remote_bytes",
        "staging_bytes",
        "net_saved_bytes",
        "selected_tensors",
        "selected_layers",
    ):
        if key in layout:
            row[key] = str(layout[key])

    if "net_saved_bytes" in layout:
        row["net_saved_source"] = "measured_vmm_layout"

    if (
        case.scheme == "native_cpu_layers"
        and status == "ok"
        and row["gpu_process_peak_mib"]
    ):
        process_peak_mib = float(row["gpu_process_peak_mib"])
        native_start = int(
            manifest["sweep"].get(
                "native_ngl_start", manifest["model"].get("gpu_layers", 61)
            )
        )
        if case.native_ngl == native_start:
            reference_peak_mib = process_peak_mib
            source = "measured_same_context_all_local_peak_delta"
        elif native_reference_peak_mib is not None:
            reference_peak_mib = native_reference_peak_mib
            source = "measured_same_context_all_local_peak_delta"
        else:
            reference_peak_mib = calibrated_peak_mib(manifest, case.context_tokens)
            source = "inferred_calibrated_all_local_peak_delta"

        # Native whole-layer offload has no meaningful VMM-style gross byte
        # count.  Capacity is the reduction in total process VRAM relative to
        # an all-local process at the identical context.  Above physical VRAM,
        # that unavailable reference is explicitly marked as inferred.
        saved_mib = max(0.0, reference_peak_mib - process_peak_mib)
        row["net_saved_bytes"] = str(round(saved_mib * MIB))
        row["net_saved_source"] = source
        row["all_local_reference_peak_mib"] = f"{reference_peak_mib:.3f}"

    gross_bytes = int(row["gross_remote_bytes"]) if row["gross_remote_bytes"] else 0
    staging_bytes = int(row["staging_bytes"]) if row["staging_bytes"] else 0
    net_bytes = int(row["net_saved_bytes"]) if row["net_saved_bytes"] else 0
    if gross_bytes:
        row["gross_remote_mib"] = f"{gross_bytes / MIB:.3f}"
        row["gross_remote_pp"] = f"{100 * gross_bytes / model_bytes:.6f}"
    if staging_bytes:
        row["staging_mib"] = f"{staging_bytes / MIB:.3f}"
    if row["net_saved_bytes"]:
        row["net_saved_mib"] = f"{net_bytes / MIB:.3f}"
        row["net_saved_pp"] = f"{100 * net_bytes / model_bytes:.6f}"

    if row["gpu_process_peak_mib"]:
        logical = float(row["gpu_process_peak_mib"]) + net_bytes / MIB
        row["logical_working_set_mib"] = f"{logical:.3f}"

    effective_baseline = baseline
    is_initial_control = (
        case.scheme != "native_cpu_layers"
        and case.target_offload_pp == Decimal(0)
    ) or (
        case.scheme == "native_cpu_layers"
        and case.native_ngl == int(manifest["sweep"].get("native_ngl_start", 61))
    )
    if status == "ok" and effective_baseline is None and is_initial_control:
        effective_baseline = (
            bench["prefill_tok_s"],
            bench["decode_tok_s"],
            canonical_decimal(case.workset_gib),
        )
    if status == "ok" and effective_baseline is not None:
        baseline_pp, baseline_tg, baseline_workset = effective_baseline
        row["baseline_source_workset_gib"] = baseline_workset
        row["prefill_retention"] = f"{float(row['prefill_tok_s']) / baseline_pp:.6f}"
        row["decode_retention"] = f"{float(row['decode_tok_s']) / baseline_tg:.6f}"
    return row


def read_resume_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "scheme",
            "workset_target_gib",
            "target_offload_pp",
            "native_ngl",
            "status",
            "checksum_status",
            "checksum_sequence_json",
            "checksum_baseline_sha256",
            "nonfinite_logits",
            "net_saved_source",
            "all_local_reference_peak_mib",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"resume CSV has incompatible header: {reader.fieldnames}")
        for row in reader:
            control = (
                f"ngl:{row['native_ngl']}"
                if row["scheme"] == "native_cpu_layers"
                else f"pp:{canonical_decimal(row['target_offload_pp'] or '0')}"
            )
            key = (
                row["scheme"],
                canonical_decimal(row["workset_target_gib"]),
                control,
            )
            rows[key] = row
    return rows


def checksum_baseline_from_rows(
    manifest: dict[str, Any],
    rows: dict[tuple[str, str, str], dict[str, str]],
) -> list[tuple[str, str]] | None:
    candidates: list[tuple[Decimal, list[tuple[str, str]]]] = []
    for (scheme, workset, control), row in rows.items():
        if scheme not in {"intertidal_dma", "cuda_zero_copy"}:
            continue
        if control != "pp:0" or row.get("status") != "ok":
            continue
        if row.get("checksum_status") not in {"baseline_captured", "match"}:
            continue
        raw = row.get("checksum_sequence_json", "")
        try:
            sequence = json.loads(raw)
            identities = [
                (
                    str(item["label"]),
                    _normalize_checksum(item["checksum"], source="resume checksum"),
                )
                for item in sequence
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        candidates.append((Decimal(workset), identities))
    if not candidates:
        return None
    _, identities = min(candidates, key=lambda item: item[0])
    validate_checksum_shape(manifest, identities, source="resume baseline")
    return identities


def is_completed(row: dict[str, str] | None, retry_errors: bool) -> bool:
    if row is None:
        return False
    if retry_errors:
        return row.get("status") in TERMINAL_STATUSES
    return True


def retention_is_low(row: dict[str, str], threshold: float) -> bool:
    if row.get("status") != "ok":
        return False
    try:
        return (
            float(row["prefill_retention"]) <= threshold
            or float(row["decode_retention"]) <= threshold
        )
    except (KeyError, ValueError):
        return False


def frontier_case_satisfies(row: dict[str, str], required_mib: float) -> bool:
    if row.get("status") != "ok":
        return False
    if required_mib <= 0:
        return True
    try:
        # Rounded CSV accounting is conservative within half a KiB-equivalent;
        # allow only a millimegabyte display-rounding tolerance.
        return float(row["net_saved_mib"]) + 0.001 >= required_mib
    except (KeyError, ValueError):
        return False


def crossing_target(
    rows: dict[tuple[str, str, str], dict[str, str]],
    scheme: str,
    workset: Decimal,
    targets: Iterable[Decimal],
    threshold: float,
    consecutive: int,
) -> Decimal | None:
    count = 0
    for target in targets:
        key = (scheme, canonical_decimal(workset), f"pp:{canonical_decimal(target)}")
        row = rows.get(key)
        if row is None:
            count = 0
            continue
        if retention_is_low(row, threshold):
            count += 1
            if count >= consecutive:
                return target
        elif row.get("status") == "ok":
            count = 0
        else:
            # OOM before enough pages are freed is neither a speed sample nor a
            # reason to stop scanning toward larger offload budgets.
            count = 0
    return None


class SweepRunner:
    def __init__(
        self,
        manifest: dict[str, Any],
        manifest_hash: str,
        output: Path,
        *,
        dry_run: bool,
        retry_errors: bool,
        show_telemetry: bool,
        max_cases: int | None,
        profile_mode: str | None = None,
        profile_output: Path | None = None,
        profile_raw: bool = False,
    ) -> None:
        self.manifest = manifest
        self.manifest_hash = manifest_hash
        self.output = output
        self.dry_run = dry_run
        self.retry_errors = retry_errors
        self.show_telemetry = show_telemetry
        self.max_cases = max_cases
        self.profile = profiling_config(manifest, profile_mode)
        if profile_raw:
            self.profile["raw_sidecar"] = True
        self.profile_output = (
            profile_output
            if profile_output is not None
            else output.with_name(f"{output.stem}.profile.jsonl")
        )
        self.executed = 0
        self.run_id = str(uuid.uuid4())
        self.rows = read_resume_rows(output)
        self.checksum_baseline = manifest_checksum_baseline(manifest)
        if self.checksum_baseline is None:
            self.checksum_baseline = checksum_baseline_from_rows(manifest, self.rows)
        self.planned_keys = set(self.rows)
        self.output_stream: TextIO | None = None
        self.writer: csv.DictWriter | None = None

    def __enter__(self) -> "SweepRunner":
        if not self.dry_run:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            exists = self.output.exists() and self.output.stat().st_size > 0
            self.output_stream = self.output.open("a", newline="")
            self.writer = csv.DictWriter(self.output_stream, fieldnames=CSV_FIELDS)
            if not exists:
                self.writer.writeheader()
                self.output_stream.flush()
        return self

    def __exit__(self, *_: object) -> None:
        if self.output_stream is not None:
            self.output_stream.close()

    def reached_limit(self) -> bool:
        return self.max_cases is not None and self.executed >= self.max_cases

    def run_case(self, case: Case) -> dict[str, str] | None:
        existing = self.rows.get(case.key())
        if is_completed(existing, self.retry_errors):
            print(
                f"skip {case.scheme} ws={case.workset_gib:.1f} "
                f"control={case.key()[2]} status={existing.get('status')}",
                flush=True,
            )
            return existing
        if self.dry_run and case.key() in self.planned_keys:
            return None
        if self.reached_limit():
            return None

        env, command = command_for_case(self.manifest, case)
        enable_backend_profile(env, self.profile)
        wrapper = remote_wrapper(
            env,
            command,
            float(self.profile["interval_s"]),
            str(self.manifest["remote"].get("gpu_id", 0)),
            self.profile,
        )
        print(
            f"case scheme={case.scheme} phase={case.phase} ws={case.workset_gib:.1f}GiB "
            f"ctx={case.context_tokens} control={case.key()[2]}",
            flush=True,
        )
        displayed_command = [
            "env",
            "-u",
            "LD_PRELOAD",
            *[f"{key}={value}" for key, value in sorted(env.items())],
            *command,
        ]
        print(f"remote command: {shell_join(displayed_command)}", flush=True)

        self.executed += 1
        if self.dry_run:
            self.planned_keys.add(case.key())
            return None

        result = execute_wrapper(
            self.manifest,
            wrapper,
            timeout_s=float(self.manifest["sweep"].get("case_timeout_s", 1800)),
            show_telemetry=self.show_telemetry,
        )
        baseline = baseline_from_rows(self.rows, case.scheme)
        native_reference_peak_mib = (
            native_all_local_peak_from_rows(
                self.manifest, self.rows, case.context_tokens
            )
            if case.scheme == "native_cpu_layers"
            else None
        )
        base_status, _, _ = classify_result(result)
        correctness = evaluate_correctness(
            self.manifest,
            case,
            result,
            base_status,
            self.checksum_baseline,
        )
        if correctness.status == "baseline_captured":
            self.checksum_baseline = correctness.baseline
        row = row_for_result(
            self.manifest,
            self.manifest_hash,
            self.run_id,
            case,
            env,
            command,
            result,
            baseline,
            correctness,
            native_reference_peak_mib,
        )
        assert self.writer is not None and self.output_stream is not None
        self.writer.writerow(row)
        self.output_stream.flush()
        os.fsync(self.output_stream.fileno())
        self.rows[case.key()] = row
        if self.profile["enabled"]:
            append_profile_sidecar(
                self.profile_output,
                profile_sidecar_record(
                    profile=self.profile,
                    run_id=self.run_id,
                    manifest_sha256=self.manifest_hash,
                    case_key={
                        "scheme": case.scheme,
                        "phase": case.phase,
                        "workset_gib": canonical_decimal(case.workset_gib),
                        "context_tokens": case.context_tokens,
                        "control": case.key()[2],
                        "page_budget": case.page_budget,
                        "native_ngl": case.native_ngl,
                    },
                    status=row["status"],
                    result=result,
                    command_identity={"env": env, "argv": command},
                ),
            )
        print(
            f"result status={row['status']} pp={row['prefill_tok_s'] or '-'} "
            f"tg={row['decode_tok_s'] or '-'} peak={row['gpu_process_peak_mib'] or '-'}MiB",
            flush=True,
        )
        if row["status"] == "correctness_fail":
            raise RuntimeError(
                f"correctness failure in {case.scheme} {case.key()[2]}: "
                f"{row['error_tail']}"
            )
        if row["status"] == "transport_error":
            raise RuntimeError(
                f"SSH transport failure in {case.scheme} {case.key()[2]}: "
                f"{row['error_tail']}"
            )
        return row

    def run_hybrid_curve(self, scheme: str, workset: Decimal) -> None:
        sweep = self.manifest["sweep"]
        threshold = float(sweep.get("stop_retention", 0.10))
        consecutive = int(sweep.get("stop_consecutive", 2))
        coarse = pp_targets(self.manifest, dense=False)

        coarse_stop = crossing_target(
            self.rows, scheme, workset, coarse, threshold, consecutive
        )
        if coarse_stop is None:
            for target in coarse:
                row = self.run_case(
                    build_case(
                        self.manifest,
                        scheme,
                        "coarse",
                        workset,
                        target_pp=target,
                    )
                )
                if self.reached_limit():
                    return
                if row is not None and row.get("status") == "unsupported":
                    return
                coarse_stop = crossing_target(
                    self.rows, scheme, workset, coarse, threshold, consecutive
                )
                if coarse_stop is not None:
                    break
        if coarse_stop is None:
            coarse_stop = coarse[-1]

        dense_step = Decimal(str(sweep.get("dense_step_pp", "0.1")))
        dense = decimal_range(Decimal(0), coarse_stop, dense_step)
        for target in dense:
            row = self.run_case(
                build_case(
                    self.manifest,
                    scheme,
                    "dense",
                    workset,
                    target_pp=target,
                )
            )
            if self.reached_limit():
                return
            if row is not None and row.get("status") == "unsupported":
                return
            if crossing_target(
                self.rows, scheme, workset, dense, threshold, consecutive
            ) is not None:
                print(
                    f"stop {scheme} ws={workset:.1f}: {consecutive} consecutive "
                    f"points at or below {threshold:.1%}",
                    flush=True,
                )
                return

    def run_native_curve(self, workset: Decimal) -> None:
        sweep = self.manifest["sweep"]
        start = int(sweep.get("native_ngl_start", self.manifest["model"].get("gpu_layers", 61)))
        stop = int(sweep.get("native_ngl_stop", 0))
        threshold = float(sweep.get("stop_retention", 0.10))
        consecutive_needed = int(sweep.get("stop_consecutive", 2))
        consecutive_low = 0

        for ngl in range(start, stop - 1, -1):
            row = self.run_case(
                build_case(
                    self.manifest,
                    "native_cpu_layers",
                    "native",
                    workset,
                    native_ngl=ngl,
                )
            )
            if self.reached_limit():
                return
            if row is None:
                continue
            if retention_is_low(row, threshold):
                consecutive_low += 1
                if consecutive_low >= consecutive_needed:
                    print(
                        f"stop native ws={workset:.1f}: {consecutive_needed} consecutive "
                        f"points at or below {threshold:.1%}",
                        flush=True,
                    )
                    return
            elif row.get("status") == "ok":
                consecutive_low = 0

    def run_hybrid_frontier(self, scheme: str, workset: Decimal) -> None:
        targets = frontier_pp_targets(self.manifest, workset)
        if not targets:
            print(
                f"frontier {scheme} ws={workset:.1f}: theoretical lower bound exceeds "
                "maximum offload",
                flush=True,
            )
            return

        required_mib = frontier_required_mib(self.manifest, workset)
        confirm_steps = int(self.manifest["sweep"].get("frontier_confirm_steps", 1))
        if (
            targets[0] > 0
            and self.manifest["sweep"].get("frontier_probe_all_local", True)
        ):
            # This is a control, not part of the monotonic fit search: it proves
            # that the identical context is beyond the all-local capacity cliff.
            self.run_case(
                build_case(
                    self.manifest,
                    scheme,
                    "capacity_all_local_control",
                    workset,
                    target_pp=Decimal(0),
                )
            )
            if self.reached_limit():
                return
        for index, target in enumerate(targets):
            row = self.run_case(
                build_case(
                    self.manifest,
                    scheme,
                    "capacity_frontier",
                    workset,
                    target_pp=target,
                )
            )
            if self.reached_limit():
                return
            if self.dry_run:
                # The remaining candidates are conditional on this point being
                # OOM or saving less VRAM than the calibrated requirement.
                if index == confirm_steps:
                    print(
                        f"frontier {scheme} ws={workset:.1f}: later 0.1pp candidates "
                        "are conditional on OOM/insufficient net saving",
                        flush=True,
                    )
                    return
                continue
            if row is None:
                continue
            if row.get("status") in {"unsupported", "correctness_fail"}:
                return
            if not frontier_case_satisfies(row, required_mib):
                continue

            # The first satisfying point is the measured capacity frontier.
            # Record the next denser point as a one-step confirmation, without
            # turning each workset into another full performance curve.
            for offset in range(1, confirm_steps + 1):
                confirm_index = index + offset
                if confirm_index >= len(targets):
                    break
                self.run_case(
                    build_case(
                        self.manifest,
                        scheme,
                        "capacity_confirm",
                        workset,
                        target_pp=targets[confirm_index],
                    )
                )
                if self.reached_limit():
                    return
            return

    def run_native_frontier(self, workset: Decimal) -> None:
        sweep = self.manifest["sweep"]
        start = int(sweep.get("native_ngl_start", self.manifest["model"].get("gpu_layers", 61)))
        stop = int(sweep.get("native_ngl_stop", 0))
        confirm_steps = int(sweep.get("frontier_confirm_steps", 1))
        required_mib = frontier_required_mib(self.manifest, workset)
        values = list(range(start, stop - 1, -1))
        for index, ngl in enumerate(values):
            row = self.run_case(
                build_case(
                    self.manifest,
                    "native_cpu_layers",
                    "capacity_frontier",
                    workset,
                    native_ngl=ngl,
                )
            )
            if self.reached_limit():
                return
            if self.dry_run:
                if index == confirm_steps:
                    print(
                        f"frontier native ws={workset:.1f}: lower -ngl values are "
                        "conditional on OOM/insufficient measured-or-inferred net saving",
                        flush=True,
                    )
                    return
                continue
            if row is None:
                continue
            if row.get("status") in {"unsupported", "correctness_fail"}:
                return
            if not frontier_case_satisfies(row, required_mib):
                continue
            for offset in range(1, confirm_steps + 1):
                confirm_index = index + offset
                if confirm_index >= len(values):
                    break
                self.run_case(
                    build_case(
                        self.manifest,
                        "native_cpu_layers",
                        "capacity_confirm",
                        workset,
                        native_ngl=values[confirm_index],
                    )
                )
                if self.reached_limit():
                    return
            return

    def run(self) -> None:
        schemes = self.manifest["sweep"].get("schemes", list(SCHEMES))
        phases = self.manifest["sweep"].get("phases", list(PHASES))
        worksets = workset_values(self.manifest)

        if "pure_curve" in phases:
            pure_workset = Decimal(
                str(self.manifest["sweep"].get("pure_curve_workset_gib", 31))
            )
            for scheme in schemes:
                if self.reached_limit():
                    return
                if scheme == "native_cpu_layers":
                    self.run_native_curve(pure_workset)
                else:
                    self.run_hybrid_curve(scheme, pure_workset)

        if "capacity_frontier" in phases:
            for scheme in schemes:
                for workset in worksets:
                    if self.reached_limit():
                        return
                    if scheme == "native_cpu_layers":
                        self.run_native_frontier(workset)
                    else:
                        self.run_hybrid_frontier(scheme, workset)


def plan_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    worksets = workset_values(manifest)
    coarse = pp_targets(manifest, dense=False)
    dense = pp_targets(manifest, dense=True)
    pure_workset = Decimal(str(manifest["sweep"].get("pure_curve_workset_gib", 31)))
    frontier = [
        {
            "target_gib": float(value),
            "context_tokens": context_for_workset(manifest, value),
            "calibrated_peak_mib": calibrated_peak_mib(
                manifest, context_for_workset(manifest, value)
            ),
            "required_net_saved_mib": frontier_required_mib(manifest, value),
            "hybrid_start_pp": float(frontier_start_pp(manifest, value)),
            "hybrid_conditional_candidates": len(frontier_pp_targets(manifest, value)),
        }
        for value in worksets
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "transport": manifest["remote"].get("transport", "ssh"),
        "target": manifest["remote"].get("target", ""),
        "schemes": manifest["sweep"].get("schemes", list(SCHEMES)),
        "phases": manifest["sweep"].get("phases", list(PHASES)),
        "pure_curve_workset_gib": float(pure_workset),
        "capacity_frontier": frontier,
        "coarse_points_per_hybrid_curve": len(coarse),
        "dense_points_per_hybrid_curve_before_early_stop": len(dense),
        "maximum_unique_points_per_full_hybrid_curve": len(dense),
        "native_points_per_curve": int(
            manifest["sweep"].get("native_ngl_start", manifest["model"].get("gpu_layers", 61))
        )
        - int(manifest["sweep"].get("native_ngl_stop", 0))
        + 1,
        "stop_rule": {
            "retention": float(manifest["sweep"].get("stop_retention", 0.10)),
            "consecutive": int(manifest["sweep"].get("stop_consecutive", 2)),
        },
        "frontier_rule": {
            "dense_step_pp": float(manifest["sweep"].get("dense_step_pp", 0.1)),
            "confirm_steps_after_first_fit": int(
                manifest["sweep"].get("frontier_confirm_steps", 1)
            ),
            "default_fit_mib": float(manifest["worksets"]["default_fit_mib"]),
            "probe_all_local_at_each_over_capacity_workset": bool(
                manifest["sweep"].get("frontier_probe_all_local", True)
            ),
            "later_points_are_conditional": True,
        },
        "correctness_gate": {
            "enabled": correctness_enabled(manifest),
            "expected_sequence_labels": expected_checksum_labels(manifest)
            if correctness_enabled(manifest)
            else [],
            "strict_checksum_schemes": manifest.get("correctness", {}).get(
                "strict_checksum_schemes",
                ["intertidal_dma", "cuda_zero_copy"],
            ),
            "require_nonfinite_field": bool(
                manifest.get("correctness", {}).get("require_nonfinite_field", True)
            ),
            "baseline_source": "manifest"
            if manifest_checksum_baseline(manifest) is not None
            else "first_successful_all_local_control",
            "failure_action": "append correctness_fail row, then abort sweep",
        },
        "remote_files_created": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/gemma-capacity-sweep.csv"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print cases without launching ssh or writing the result CSV",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="stop after this many new cases (also limits dry-run output)",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="rerun non-terminal error/timeout/parse_error rows while resuming",
    )
    parser.add_argument("--show-telemetry", action="store_true")
    parser.add_argument(
        "--profile",
        choices=PROFILE_MODES,
        help="override profiling mode: off, light system counters, or deep backend events",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="local JSONL sidecar (default: <output stem>.profile.jsonl)",
    )
    parser.add_argument(
        "--profile-raw",
        action="store_true",
        help="include raw telemetry samples and backend records in each JSONL case",
    )
    parser.add_argument(
        "--profile-force-direct",
        action="store_true",
        help=(
            "disable CUDA Graphs for selected light-profile cases without enabling "
            "CUDA timing events (observer-overhead control)"
        ),
    )
    parser.add_argument(
        "--profile-backend-off",
        action="store_true",
        help=(
            "keep deep-mode system telemetry/direct execution but disable CUDA "
            "timing events (matched observer-overhead control)"
        ),
    )
    parser.add_argument(
        "--profile-no-pcie-dmon",
        action="store_true",
        help="disable the optional one-second nvidia-smi dmon PCIe fallback",
    )
    parser.add_argument(
        "--only-offload-pp",
        help=(
            "run only these comma-separated hybrid offload percentages, e.g. "
            "0,3.2,10,22.4; intended for selected profiling probes"
        ),
    )
    parser.add_argument(
        "--schemes",
        help="comma-separated scheme override",
    )
    parser.add_argument(
        "--profile-workset-gib",
        type=Decimal,
        help="workset for --only-offload-pp (default: pure_curve_workset_gib)",
    )
    parser.add_argument(
        "--profile-benchmark",
        choices=("both", "prefill", "decode"),
        default="both",
        help=(
            "selected-probe workload; use separate prefill/decode invocations "
            "to keep backend timing phases unambiguous"
        ),
    )
    parser.add_argument(
        "--phases",
        help="comma-separated override: pure_curve,capacity_frontier",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, manifest_hash = load_manifest(args.manifest)
    if args.phases:
        phases = [item.strip() for item in args.phases.split(",") if item.strip()]
        unknown = set(phases) - set(PHASES)
        if unknown:
            raise ValueError(f"unknown phases: {sorted(unknown)}")
        manifest["sweep"]["phases"] = phases
    if args.schemes:
        schemes = [item.strip() for item in args.schemes.split(",") if item.strip()]
        unknown_schemes = set(schemes) - set(SCHEMES)
        if unknown_schemes:
            raise ValueError(f"unknown schemes: {sorted(unknown_schemes)}")
        manifest["sweep"]["schemes"] = schemes
    if args.profile_force_direct:
        manifest.setdefault("profiling", {})["force_direct"] = True
    if args.profile_backend_off:
        if args.profile != "deep":
            raise ValueError("--profile-backend-off requires --profile deep")
        manifest.setdefault("profiling", {})["backend_enabled"] = False
    if args.profile_no_pcie_dmon:
        manifest.setdefault("profiling", {})[f"{args.profile or 'light'}_pcie_dmon"] = False
    if args.profile_benchmark != "both" and not args.only_offload_pp:
        raise ValueError("--profile-benchmark requires --only-offload-pp")
    if args.profile_benchmark == "prefill":
        manifest["benchmark"]["generation"] = 0
    elif args.profile_benchmark == "decode":
        manifest["benchmark"]["prompt"] = 0

    selected_targets: list[Decimal] | None = None
    if args.only_offload_pp:
        selected_targets = []
        for raw in args.only_offload_pp.split(","):
            if not raw.strip():
                continue
            target = Decimal(raw.strip())
            if target < 0 or target > Decimal(100):
                raise ValueError("--only-offload-pp values must be between 0 and 100")
            if target not in selected_targets:
                selected_targets.append(target)
        if not selected_targets:
            raise ValueError("--only-offload-pp did not contain a percentage")
        selected_schemes = manifest["sweep"].get("schemes", list(SCHEMES))
        if any(scheme == "native_cpu_layers" for scheme in selected_schemes):
            raise ValueError("--only-offload-pp supports hybrid schemes only")
        # A fresh selected-case output needs an all-local checksum reference.
        # Keep it first even if the caller lists targets out of order.
        selected_targets = [Decimal(0)] + [
            target for target in selected_targets if target != 0
        ]
    print(json.dumps(plan_summary(manifest), indent=2, sort_keys=True), flush=True)
    with SweepRunner(
        manifest,
        manifest_hash,
        args.output,
        dry_run=args.dry_run,
        retry_errors=args.retry_errors,
        show_telemetry=args.show_telemetry,
        max_cases=args.max_cases,
        profile_mode=args.profile,
        profile_output=args.profile_output,
        profile_raw=args.profile_raw,
    ) as runner:
        if selected_targets is None:
            runner.run()
        else:
            workset = args.profile_workset_gib or Decimal(
                str(manifest["sweep"].get("pure_curve_workset_gib", 31))
            )
            for scheme in manifest["sweep"].get("schemes", list(SCHEMES)):
                for target in selected_targets:
                    if runner.reached_limit():
                        break
                    runner.run_case(
                        build_case(
                            manifest,
                            scheme,
                            f"profile_probe_{args.profile_benchmark}",
                            workset,
                            target_pp=target,
                        )
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
