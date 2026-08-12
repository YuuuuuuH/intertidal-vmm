#!/usr/bin/env python3
"""Resumable populated-context capacity benchmark for Gemma-4 31B Q8.

The runner stays local.  It streams a shell wrapper to the RTX 5090 over SSH,
parses all output locally, and creates no remote scripts, logs, or result data.

Unlike the Q4 allocation sweep, a successful capacity point must execute one
combined ``llama-bench -pg P,G`` test.  That command prefills P tokens and then
decodes G tokens in the same context, so all C=P+G KV positions are genuinely
populated.  ``--ctx-size C`` is only the matching allocation ceiling.

Order is fixed: stable-download gate, all-local load probe, coarse staging
capacity curve, coarse zero-copy/native curves, then page-exact VMM refinement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

import gemma_capacity_sweep as base


SCHEMA_VERSION = 1
MIB = 1 << 20

DIRECT = "all_local_direct"
DMA = "intertidal_dma"
ZERO_COPY = "cuda_zero_copy"
NATIVE = "native_cpu_layers"
VMM_SCHEMES = (DMA, ZERO_COPY)
SCHEMES = (DIRECT, DMA, ZERO_COPY, NATIVE)

PHASE_DIRECT = "direct_load"
PHASE_STAGING_COARSE = "staging_coarse"
PHASE_ALTERNATES = "alternate_coarse"
PHASE_FINE = "fine_0_1pp"
PHASE_CONFIRM = "threshold_confirmation"
PHASE_REFERENCE = "checksum_reference"
PHASES = (PHASE_DIRECT, PHASE_STAGING_COARSE, PHASE_ALTERNATES, PHASE_FINE)

SCREEN_PROFILE = "screen"
CONFIRM_PROFILE = "confirm"
MODEL_PREFIX = "__GEMMA_Q8_MODEL__|"
MODEL_HASH_PREFIX = "__GEMMA_Q8_SHA256__|"
TERMINAL_STATUSES = {
    "ok", "oom", "cuda_capacity_fail", "unsupported", "correctness_fail"
}

PHASE_TIMING_PREFIX = "llama_bench_phase_timing,"
PHASE_TIMING_RE = re.compile(
    r"llama_bench_phase_timing,"
    r"phase=(?P<phase>prompt|decode),"
    r"rep=(?P<rep>[1-9][0-9]*),"
    r"tokens=(?P<tokens>[1-9][0-9]*),"
    r"ns=(?P<ns>[1-9][0-9]*)"
)


CSV_FIELDS = [
    "schema_version",
    "run_id",
    "case_time_utc",
    "manifest_sha256",
    "experiment_id",
    "scheme",
    "phase",
    "profile",
    "probe_kind",
    "control_key",
    "target_offload_pp",
    "page_budget",
    "native_ngl",
    "context_tokens",
    "actual_context_tokens",
    "reserved_context_tokens",
    "filled_context_tokens",
    "prompt_tokens",
    "generation_tokens",
    "repetitions",
    "total_prompt_eval_tokens",
    "total_decode_eval_tokens",
    "populated_context_proven",
    "status",
    "exit_code",
    "elapsed_s",
    "expected_outcome",
    "expectation_met",
    "model_file_bytes",
    "eligible_tensor_bytes",
    "gguf_tensor_count",
    "model_context_length",
    "model_metadata_sha256",
    "model_sha256",
    "offload_budget_basis",
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
    "selected_tensors",
    "selected_layers",
    "native_cpu_model_mib",
    "native_cuda_model_mib",
    "perf_prompt_ms",
    "perf_decode_ms",
    "prefill_tok_s",
    "decode_tok_s",
    "is_scheme_origin",
    "is_global_origin",
    "scheme_origin_scheme",
    "scheme_origin_control",
    "scheme_origin_context_tokens",
    "prefill_retention_vs_scheme_origin",
    "decode_retention_vs_scheme_origin",
    "global_origin_scheme",
    "global_origin_control",
    "global_origin_context_tokens",
    "prefill_retention_vs_global_origin",
    "decode_retention_vs_global_origin",
    "checksum_status",
    "checksum_reference_scheme",
    "checksum_sequence_json",
    "checksum_reference_sha256",
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
    "remote_artifacts_created",
    "error_tail",
]

FRONTIER_FIELDS = [
    "schema_version",
    "scheme",
    "control_key",
    "target_offload_pp",
    "native_ngl",
    "frontier_phase",
    "frontier_context_tokens",
    "actual_context_tokens",
    "minimum_page_budget",
    "last_failed_page_budget",
    "first_failed_context_tokens",
    "prompt_tokens",
    "generation_tokens",
    "prefill_tok_s",
    "decode_tok_s",
    "prefill_retention_vs_scheme_origin",
    "decode_retention_vs_scheme_origin",
    "prefill_retention_vs_global_origin",
    "decode_retention_vs_global_origin",
    "confirmed_prefill_tok_s",
    "confirmed_decode_tok_s",
    "confirmed_prefill_retention_vs_global_origin",
    "confirmed_decode_retention_vs_global_origin",
    "gross_remote_mib",
    "net_saved_mib",
    "gpu_process_peak_mib",
    "gpu_device_peak_mib",
    "checksum_status",
]


GGUF_METADATA_PROBE = r'''import hashlib
import math
import os
import re
import struct
import sys

path = sys.argv[1]
eligible_re = re.compile(
    r"^blk[.][0-9]+[.](?:attn_(?:k|v|q|output)|ffn_(?:down|gate|up))[.]weight$"
)
fixed = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
    10: 8, 11: 8, 12: 8,
}
traits = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98),
    19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1),
    25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2),
}

with open(path, "rb") as stream:
    def read_exact(n):
        data = stream.read(n)
        if len(data) != n:
            raise EOFError("truncated GGUF metadata")
        return data

    def scalar(fmt):
        return struct.unpack("<" + fmt, read_exact(struct.calcsize(fmt)))[0]

    def string(decode=True):
        length = scalar("Q")
        if length > (1 << 31):
            raise ValueError("unreasonable GGUF string length")
        raw = read_exact(length)
        return raw.decode("utf-8") if decode else raw

    def value(kind, keep=False):
        if kind in fixed:
            raw = read_exact(fixed[kind])
            if not keep:
                return None
            formats = {
                0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
            }
            return struct.unpack("<" + formats[kind], raw)[0]
        if kind == 8:
            text = string(decode=keep)
            return text if keep else None
        if kind == 9:
            element_kind = scalar("I")
            count = scalar("Q")
            if count > (1 << 32):
                raise ValueError("unreasonable GGUF array length")
            for _ in range(count):
                value(element_kind, False)
            return None
        raise ValueError("unknown GGUF metadata type %d" % kind)

    if read_exact(4) != b"GGUF":
        raise ValueError("bad GGUF magic")
    version = scalar("I")
    if version not in (2, 3):
        raise ValueError("unsupported GGUF version %d" % version)
    tensor_count = scalar("Q")
    kv_count = scalar("Q")
    if tensor_count <= 0 or tensor_count > 1000000 or kv_count > 1000000:
        raise ValueError("unreasonable GGUF counts")
    alignment = 32
    architecture = ""
    context_length = 0
    for _ in range(kv_count):
        key = string()
        kind = scalar("I")
        keep = key == "general.alignment" or key == "general.architecture" or key.endswith(".context_length")
        kept = value(kind, keep)
        if key == "general.alignment":
            alignment = int(kept)
        elif key == "general.architecture":
            architecture = str(kept)
        elif key.endswith(".context_length"):
            context_length = max(context_length, int(kept))
    if alignment <= 0 or alignment > (1 << 20):
        raise ValueError("invalid GGUF alignment")

    eligible_bytes = 0
    offsets = []
    for _ in range(tensor_count):
        name = string()
        n_dims = scalar("I")
        if n_dims <= 0 or n_dims > 4:
            raise ValueError("invalid tensor rank")
        dims = [scalar("Q") for _ in range(n_dims)]
        ggml_type = scalar("I")
        offset = scalar("Q")
        offsets.append(offset)
        if eligible_re.fullmatch(name):
            if ggml_type not in traits:
                raise ValueError("unknown eligible tensor type %d" % ggml_type)
            block, type_size = traits[ggml_type]
            row_bytes = ((dims[0] + block - 1) // block) * type_size
            eligible_bytes += row_bytes * math.prod(dims[1:])

    data_start = ((stream.tell() + alignment - 1) // alignment) * alignment
    file_size = os.fstat(stream.fileno()).st_size
    if data_start >= file_size or not offsets or max(offsets) >= file_size - data_start:
        raise ValueError("GGUF tensor data offsets exceed file")
    if eligible_bytes <= 0:
        raise ValueError("no eligible transformer matrix tensors")
    if not architecture or context_length <= 0:
        raise ValueError("GGUF architecture/context length metadata missing")
    stream.seek(0)
    metadata_sha256 = hashlib.sha256(read_exact(data_start)).hexdigest()
    print("1|%d|%d|%d|%s" % (tensor_count, eligible_bytes, context_length, metadata_sha256))
'''


@dataclass(frozen=True)
class ModelInfo:
    file_bytes: int
    eligible_tensor_bytes: int
    tensor_count: int
    context_length: int
    metadata_sha256: str
    file_sha256: str
    mtime_epoch: int


@dataclass(frozen=True)
class Profile:
    name: str
    generation_tokens: int
    repetitions: int


@dataclass(frozen=True)
class Q8Case:
    scheme: str
    phase: str
    profile: str
    probe_kind: str
    context_tokens: int
    target_offload_pp: Decimal | None = None
    page_budget: int | None = None
    native_ngl: int | None = None

    def control_key(self) -> str:
        if self.scheme == NATIVE:
            return f"ngl:{self.native_ngl}"
        if self.scheme in VMM_SCHEMES:
            return f"pages:{self.page_budget or 0}"
        return "local"

    def key(self) -> tuple[str, str, str, int]:
        # Coarse and fine phases reuse identical physical evidence.
        return self.scheme, self.control_key(), self.profile, self.context_tokens


@dataclass(frozen=True)
class ParsedPerformance:
    actual_context_tokens: int
    prompt_ms: float
    prompt_samples: int
    prefill_tok_s: float
    decode_ms: float
    decode_samples: int
    decode_tok_s: float


@dataclass(frozen=True)
class CorrectnessEvaluation:
    status: str
    entries: tuple[base.LogitChecksum, ...] = ()
    reference_scheme: str = ""
    reference_sha256: str = ""
    nonfinite_logits: int | None = None
    error: str = ""


def canonical_decimal(value: Decimal | str | float) -> str:
    return base.canonical_decimal(value)


def decimal_range(start: Decimal, stop: Decimal, step: Decimal) -> list[Decimal]:
    return base.decimal_range(start, stop, step)


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    validate_manifest(manifest)
    return manifest, hashlib.sha256(raw).hexdigest()


def profile_for(manifest: dict[str, Any], name: str) -> Profile:
    try:
        raw = manifest["benchmark"]["profiles"][name]
    except KeyError as exc:
        raise ValueError(f"benchmark.profiles.{name} is required") from exc
    return Profile(name, int(raw["generation"]), int(raw["repetitions"]))


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for section in ("remote", "paths", "model", "download", "benchmark", "context", "sweep"):
        if section not in manifest:
            raise ValueError(f"manifest is missing section {section!r}")
    remote = manifest["remote"]
    if remote.get("transport", "ssh") not in {"ssh", "local"}:
        raise ValueError("remote.transport must be 'ssh' or 'local'")
    if remote.get("transport", "ssh") == "ssh" and not remote.get("target"):
        raise ValueError("remote.target is required for SSH transport")
    for key in ("model", "hybrid_bench", "native_bench"):
        if not manifest["paths"].get(key):
            raise ValueError(f"paths.{key} is required")
    if int(manifest["model"].get("page_bytes", 0)) <= 0:
        raise ValueError("model.page_bytes must be positive")
    if manifest["model"].get("offload_basis", "eligible_tensor_bytes") != "eligible_tensor_bytes":
        raise ValueError("model.offload_basis must be eligible_tensor_bytes")

    download = manifest["download"]
    if int(download.get("stable_polls", 3)) < 1:
        raise ValueError("download.stable_polls must be >= 1")
    if float(download.get("poll_interval_s", 20)) <= 0:
        raise ValueError("download.poll_interval_s must be positive")
    if int(download.get("minimum_bytes", 1)) <= 0:
        raise ValueError("download.minimum_bytes must be positive")

    for name in (SCREEN_PROFILE, CONFIRM_PROFILE):
        profile = profile_for(manifest, name)
        if profile.generation_tokens <= 0 or profile.repetitions <= 0:
            raise ValueError(f"benchmark.profiles.{name} values must be positive")

    context = manifest["context"]
    quantum = int(context.get("quantum_tokens", 256))
    minimum = int(context.get("minimum_context_tokens", 512))
    maximum = int(context.get("maximum_context_tokens", 262144))
    coarse = int(context.get("coarse_step_tokens", 4096))
    fine = int(context.get("fine_step_tokens", 256))
    if min(quantum, minimum, maximum, coarse, fine) <= 0 or minimum > maximum:
        raise ValueError("context sizes and steps must be positive and ordered")
    if any(value % quantum for value in (minimum, maximum, coarse, fine)):
        raise ValueError("context sizes and steps must be multiples of quantum_tokens")
    if minimum <= max(
        profile_for(manifest, SCREEN_PROFILE).generation_tokens,
        profile_for(manifest, CONFIRM_PROFILE).generation_tokens,
    ):
        raise ValueError("minimum context must exceed every generation length")

    sweep = manifest["sweep"]
    for key, default in (("coarse_step_pp", "1"), ("dense_step_pp", "0.1"), ("max_offload_pp", "95")):
        if Decimal(str(sweep.get(key, default))) <= 0:
            raise ValueError(f"sweep.{key} must be positive")
    if Decimal(str(sweep.get("coarse_start_pp", "0"))) < 0:
        raise ValueError("sweep.coarse_start_pp must be non-negative")
    threshold = float(sweep.get("stop_retention", 0.10))
    if not 0 < threshold <= 1:
        raise ValueError("sweep.stop_retention must be in (0, 1]")
    if int(sweep.get("stop_consecutive", 1)) < 1:
        raise ValueError("sweep.stop_consecutive must be >= 1")
    if sweep.get("direct_probe_expected", "") not in {"", "ok", "oom"}:
        raise ValueError("sweep.direct_probe_expected must be '', 'ok', or 'oom'")
    fine_schemes = set(sweep.get("fine_schemes", list(VMM_SCHEMES)))
    if not fine_schemes.issubset(VMM_SCHEMES):
        raise ValueError("sweep.fine_schemes may contain only VMM schemes")
    phases = set(sweep.get("phases", list(PHASES)))
    if not phases.issubset(PHASES):
        raise ValueError(f"unknown phases: {sorted(phases - set(PHASES))}")


def model_probe_wrapper(model_path: str) -> str:
    quoted_path = shlex.quote(model_path)
    quoted_probe = shlex.quote(GGUF_METADATA_PROBE)
    return f"""#!/usr/bin/env bash
set +e
unset LD_PRELOAD
model={quoted_path}
if [[ ! -f "$model" ]]; then
    printf '{MODEL_PREFIX}0|0|0|0|0|0|0|0|-\n'
    exit 0
fi
size=$(stat -c %s -- "$model" 2>/dev/null || stat -f %z -- "$model" 2>/dev/null)
mtime=$(stat -c %Y -- "$model" 2>/dev/null || stat -f %m -- "$model" 2>/dev/null)
partial=0
if [[ -e "$model.aria2" ]]; then partial=1; fi
metadata=$(python3 -c {quoted_probe} "$model" 2>/dev/null)
meta_rc=$?
metadata_ok=0
tensor_count=0
eligible_bytes=0
context_length=0
metadata_sha256=-
if [[ $meta_rc -eq 0 && "${{metadata%%|*}}" == 1 ]]; then
    IFS='|' read -r metadata_ok tensor_count eligible_bytes context_length metadata_sha256 <<< "$metadata"
fi
printf '{MODEL_PREFIX}1|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "$size" "$mtime" "$partial" "$metadata_ok" "$tensor_count" "$eligible_bytes" \
    "$context_length" "$metadata_sha256"
"""


def parse_model_probe(text: str) -> tuple[bool, ModelInfo, bool, bool]:
    lines = [line for line in text.splitlines() if line.startswith(MODEL_PREFIX)]
    if not lines:
        raise ValueError("model probe did not emit its status record")
    fields = lines[-1].split("|")[1:]
    if len(fields) != 9:
        raise ValueError(f"malformed model probe record: {lines[-1]!r}")
    exists, size, mtime, partial, metadata_ok, tensor_count, eligible, context = (
        int(item) for item in fields[:8]
    )
    metadata_sha256 = fields[8]
    return (
        bool(exists),
        ModelInfo(size, eligible, tensor_count, context, metadata_sha256, "", mtime),
        bool(partial),
        bool(metadata_ok),
    )


def model_hash_wrapper(model_path: str) -> str:
    quoted_path = shlex.quote(model_path)
    return f"""#!/usr/bin/env bash
set +e
unset LD_PRELOAD
model={quoted_path}
if [[ ! -f "$model" ]]; then
    printf '{MODEL_HASH_PREFIX}missing\n'
    exit 1
fi
digest=$(sha256sum -- "$model" 2>/dev/null)
hash_rc=$?
digest=${{digest%%[[:space:]]*}}
if [[ $hash_rc -ne 0 || ! "$digest" =~ ^[0-9a-fA-F]{{64}}$ ]]; then
    printf '{MODEL_HASH_PREFIX}error\n'
    exit 1
fi
printf '{MODEL_HASH_PREFIX}%s\n' "$digest"
"""


def parse_model_hash(text: str) -> str:
    values = [
        line[len(MODEL_HASH_PREFIX) :].strip().lower()
        for line in text.splitlines()
        if line.startswith(MODEL_HASH_PREFIX)
    ]
    if not values or re.fullmatch(r"[0-9a-f]{64}", values[-1]) is None:
        raise ValueError("full-model SHA-256 probe failed")
    return values[-1]


def validate_model_info(manifest: dict[str, Any], info: ModelInfo) -> None:
    expected_context = int(manifest["model"].get("expected_context_length", 0))
    if expected_context and info.context_length != expected_context:
        raise ValueError(
            f"Q8 GGUF context length mismatch: expected {expected_context}, got {info.context_length}"
        )
    maximum = int(manifest["context"].get("maximum_context_tokens", 262144))
    if maximum > info.context_length:
        raise ValueError(
            f"requested maximum context {maximum} exceeds GGUF limit {info.context_length}"
        )
    if re.fullmatch(r"[0-9a-f]{64}", info.metadata_sha256) is None:
        raise ValueError("Q8 GGUF metadata SHA-256 is invalid")
    if info.file_sha256 and re.fullmatch(r"[0-9a-f]{64}", info.file_sha256) is None:
        raise ValueError("Q8 full-file SHA-256 is invalid")


def wait_for_model(
    manifest: dict[str, Any], *, dry_run: bool, skip_wait: bool
) -> ModelInfo:
    model_path = str(manifest["paths"]["model"])
    if dry_run:
        info = ModelInfo(
            int(manifest["download"].get("planning_file_bytes", manifest["download"]["minimum_bytes"])),
            int(manifest["download"].get("planning_eligible_tensor_bytes", manifest["download"]["minimum_bytes"])),
            int(manifest["download"].get("planning_tensor_count", 1)),
            int(manifest["model"].get("expected_context_length", 262144)),
            "0" * 64,
            "0" * 64,
            0,
        )
        print(
            f"model gate (dry-run): wait for stable, metadata-valid {model_path!r}; "
            f"planning eligible bytes={info.eligible_tensor_bytes}",
            flush=True,
        )
        return info

    required = 1 if skip_wait else int(manifest["download"].get("stable_polls", 3))
    poll_s = float(manifest["download"].get("poll_interval_s", 20))
    timeout_s = float(manifest["download"].get("timeout_s", 86400))
    minimum_bytes = int(manifest["download"].get("minimum_bytes", 1))
    started = time.monotonic()
    previous: ModelInfo | None = None
    stable = 0
    while True:
        result = base.execute_wrapper(
            manifest,
            model_probe_wrapper(model_path),
            timeout_s=min(180.0, timeout_s),
            show_telemetry=False,
        )
        combined = result.stdout + "\n" + result.stderr
        if result.returncode == 255 or base.TRANSPORT_ERROR_RE.search(combined):
            raise RuntimeError(
                f"model wait SSH transport failure: {base.last_error_lines(combined)}"
            )
        exists, info, partial, metadata_ok = parse_model_probe(result.stdout)
        ready = (
            exists
            and info.file_bytes >= minimum_bytes
            and not partial
            and metadata_ok
            and info.eligible_tensor_bytes > 0
        )
        if ready and info == previous:
            stable += 1
        elif ready:
            stable = 1
        else:
            stable = 0
        previous = info if ready else None
        print(
            f"Q8 download gate: exists={int(exists)} size={info.file_bytes} "
            f"partial={int(partial)} metadata={int(metadata_ok)} "
            f"eligible={info.eligible_tensor_bytes} stable={stable}/{required}",
            flush=True,
        )
        if stable >= required:
            hash_result = base.execute_wrapper(
                manifest,
                model_hash_wrapper(model_path),
                timeout_s=float(manifest["download"].get("hash_timeout_s", 3600)),
                show_telemetry=False,
            )
            hash_combined = hash_result.stdout + "\n" + hash_result.stderr
            if hash_result.returncode == 255 or base.TRANSPORT_ERROR_RE.search(hash_combined):
                raise RuntimeError(
                    f"model hash SSH transport failure: {base.last_error_lines(hash_combined)}"
                )
            file_sha256 = parse_model_hash(hash_result.stdout)
            complete = ModelInfo(
                info.file_bytes,
                info.eligible_tensor_bytes,
                info.tensor_count,
                info.context_length,
                info.metadata_sha256,
                file_sha256,
                info.mtime_epoch,
            )
            validate_model_info(manifest, complete)
            return complete
        elapsed = time.monotonic() - started
        if elapsed >= timeout_s:
            raise TimeoutError(f"Q8 model did not become ready within {timeout_s:.0f}s")
        time.sleep(min(poll_s, max(0.0, timeout_s - elapsed)))


def page_budget_for_pp(manifest: dict[str, Any], model: ModelInfo, pp: Decimal) -> int:
    pages = (
        Decimal(model.eligible_tensor_bytes)
        * pp
        / Decimal(100)
        / Decimal(int(manifest["model"]["page_bytes"]))
    ).to_integral_value(rounding=ROUND_HALF_UP)
    return max(0, int(pages))


def build_case(
    manifest: dict[str, Any],
    model: ModelInfo,
    scheme: str,
    phase: str,
    profile: str,
    probe_kind: str,
    context_tokens: int,
    *,
    target_pp: Decimal | None = None,
    page_budget: int | None = None,
    native_ngl: int | None = None,
) -> Q8Case:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    selected_profile = profile_for(manifest, profile)
    if context_tokens <= selected_profile.generation_tokens:
        raise ValueError("context must exceed generation length")
    if scheme == NATIVE:
        if native_ngl is None:
            raise ValueError("native case requires native_ngl")
        return Q8Case(
            scheme, phase, profile, probe_kind, context_tokens, native_ngl=native_ngl
        )
    if page_budget is not None and page_budget < 0:
        raise ValueError("page_budget must be non-negative")
    target = Decimal(0) if target_pp is None else target_pp
    pages = (
        page_budget
        if page_budget is not None
        else page_budget_for_pp(manifest, model, target)
    )
    realized_target = (
        Decimal(pages)
        * Decimal(int(manifest["model"]["page_bytes"]))
        * Decimal(100)
        / Decimal(model.eligible_tensor_bytes)
    )
    return Q8Case(
        scheme,
        phase,
        profile,
        probe_kind,
        context_tokens,
        target_offload_pp=realized_target,
        page_budget=(
            pages if scheme in VMM_SCHEMES else 0
        ),
    )


def command_for_case(
    manifest: dict[str, Any], case: Q8Case
) -> tuple[dict[str, str], list[str]]:
    paths = manifest["paths"]
    bench = manifest["benchmark"]
    profile = profile_for(manifest, case.profile)
    prompt = case.context_tokens - profile.generation_tokens
    binary = paths["native_bench"] if case.scheme == NATIVE else paths["hybrid_bench"]
    env = {str(key): str(value) for key, value in manifest.get("environment", {}).items()}
    env["LLAMA_BENCH_LOGIT_CHECKSUM"] = "1"
    command = [
        str(binary),
        "-m", str(paths["model"]),
        "-ngl", str(int(manifest["model"].get("gpu_layers", 61))),
        "-p", "0",
        "-n", "0",
        "-pg", f"{prompt},{profile.generation_tokens}",
        "-d", "0",
        "-r", str(profile.repetitions),
        "-t", str(int(bench.get("threads", 8))),
        "-b", str(int(bench.get("batch", 512))),
        "-ub", str(int(bench.get("ubatch", 512))),
        "-fa", str(bench.get("flash_attention", "on")),
        "-ctk", str(bench.get("cache_type_k", "f16")),
        "-ctv", str(bench.get("cache_type_v", "f16")),
        "-nkvo", "0",
        "-mmp", "1" if bench.get("mmap", True) else "0",
        "-o", "json",
        "--ctx-size", str(case.context_tokens),
        "--no-warmup",
        "-v",
    ]
    command.extend(str(arg) for arg in bench.get("extra_args", []))
    if case.scheme in VMM_SCHEMES:
        assert case.page_budget is not None
        # Keep the Hybrid override even for the 0-page control.  That records
        # layout/repack overhead separately from the stock all-local probe.
        env["GGML_CUDA_HYBRID_PAGE_BUDGET"] = str(case.page_budget)
        env["GGML_CUDA_HYBRID_MODE"] = "staging" if case.scheme == DMA else "zero_copy"
        command.extend(
            ["-ot", str(manifest["model"].get("hybrid_override", base.HYBRID_OVERRIDE))]
        )
    elif case.scheme == NATIVE:
        assert case.native_ngl is not None
        ngl_index = command.index("-ngl") + 1
        command[ngl_index] = str(case.native_ngl)
    return env, command


def parse_single_json_record(stdout: str) -> dict[str, Any]:
    start = stdout.find("[")
    stop = stdout.rfind("]")
    if start < 0 or stop < start:
        raise ValueError("llama-bench JSON array is missing")
    parsed = json.loads(stdout[start : stop + 1])
    if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], dict):
        raise ValueError("expected exactly one combined -pg JSON record")
    return parsed[0]


def parse_performance(
    manifest: dict[str, Any], case: Q8Case, result: base.CommandResult
) -> ParsedPerformance:
    profile = profile_for(manifest, case.profile)
    prompt = case.context_tokens - profile.generation_tokens
    record = parse_single_json_record(result.stdout)
    expected = {
        "n_prompt": prompt,
        "n_gen": profile.generation_tokens,
        "n_depth": 0,
    }
    for key, wanted in expected.items():
        try:
            actual = int(record[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"JSON record has invalid {key}") from exc
        if actual != wanted:
            raise ValueError(f"JSON {key} mismatch: expected {wanted}, got {actual}")
    try:
        actual_context = int(record["n_ctx"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("JSON record has invalid n_ctx") from exc
    if actual_context < case.context_tokens:
        raise ValueError(
            f"JSON n_ctx is smaller than requested: {actual_context} < {case.context_tokens}"
        )

    records: list[tuple[str, int, int, int]] = []
    for line in result.stderr.splitlines():
        if not line.startswith(PHASE_TIMING_PREFIX):
            continue
        match = PHASE_TIMING_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed phase timing line: {line!r}")
        records.append(
            (
                match.group("phase"),
                int(match.group("rep")),
                int(match.group("tokens")),
                int(match.group("ns")),
            )
        )

    expected_records = [
        (phase, rep, tokens)
        for rep in range(1, profile.repetitions + 1)
        for phase, tokens in (
            ("prompt", prompt),
            ("decode", profile.generation_tokens),
        )
    ]
    observed_records = [(phase, rep, tokens) for phase, rep, tokens, _ in records]
    if observed_records != expected_records:
        raise ValueError(
            "phase timing sequence mismatch: "
            f"expected {expected_records}, got {observed_records}"
        )

    pp_ns = sum(ns for phase, _, _, ns in records if phase == "prompt")
    tg_ns = sum(ns for phase, _, _, ns in records if phase == "decode")
    pp_n = profile.repetitions * prompt
    tg_n = profile.repetitions * profile.generation_tokens
    if pp_ns <= 0 or tg_ns <= 0:
        raise ValueError(f"non-positive aggregate phase time: prompt_ns={pp_ns}, decode_ns={tg_ns}")
    pp_ms = pp_ns / 1_000_000.0
    tg_ms = tg_ns / 1_000_000.0
    pp_tps = 1_000_000_000.0 * pp_n / pp_ns
    tg_tps = 1_000_000_000.0 * tg_n / tg_ns
    if not all(math.isfinite(value) and value > 0 for value in (pp_ms, tg_ms, pp_tps, tg_tps)):
        raise ValueError("non-finite or non-positive aggregate phase performance")
    return ParsedPerformance(actual_context, pp_ms, pp_n, pp_tps, tg_ms, tg_n, tg_tps)


def classify_result(
    manifest: dict[str, Any], case: Q8Case, result: base.CommandResult, model: ModelInfo
) -> tuple[str, ParsedPerformance | None, str]:
    combined = result.stdout + "\n" + result.stderr
    if result.timed_out:
        return "timeout", None, "case timeout"
    if result.returncode == 255 or base.TRANSPORT_ERROR_RE.search(combined):
        return "transport_error", None, base.last_error_lines(combined) or "SSH exited with status 255"
    if base.UNSUPPORTED_RE.search(combined):
        return "unsupported", None, base.last_error_lines(combined)
    if base.OOM_RE.search(combined):
        return "oom", None, base.last_error_lines(combined)
    if re.search(r"failed to create context", combined, re.IGNORECASE):
        valid_context = case.context_tokens <= model.context_length
        configured = bool(
            manifest["sweep"].get("valid_context_create_failure_is_oom", False)
        )
        if valid_context and configured:
            return "oom", None, base.last_error_lines(combined)
        return (
            "error",
            None,
            "context creation failed without CUDA allocation evidence; "
            + base.last_error_lines(combined),
        )
    if result.returncode in {-6, 134} and re.search(
        r"(?:CUDA|MMQ|ggml_cuda|GGML_ASSERT|Aborted)", combined, re.IGNORECASE
    ):
        return "cuda_capacity_fail", None, base.last_error_lines(combined)
    if result.returncode:
        return "error", None, base.last_error_lines(combined)
    try:
        perf = parse_performance(manifest, case, result)
    except (ValueError, json.JSONDecodeError) as exc:
        return "parse_error", None, f"{exc}; {base.last_error_lines(combined)}"
    return "ok", perf, ""


def expected_checksum_labels(manifest: dict[str, Any], profile_name: str) -> list[str]:
    repetitions = profile_for(manifest, profile_name).repetitions
    return [label for _ in range(repetitions) for label in ("prompt", "decode")]


def checksum_sha256(entries: Sequence[base.LogitChecksum]) -> str:
    canonical = json.dumps(
        [{"label": entry.label, "checksum": entry.checksum.lower()} for entry in entries],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def evaluate_correctness(
    manifest: dict[str, Any],
    case: Q8Case,
    result: base.CommandResult,
    status: str,
    reference: tuple[str, Sequence[base.LogitChecksum]] | None,
) -> CorrectnessEvaluation:
    if status != "ok":
        return CorrectnessEvaluation("not_run")
    entries = tuple(base.parse_logit_checksums(result.stdout + "\n" + result.stderr))
    labels = [entry.label for entry in entries]
    expected = expected_checksum_labels(manifest, case.profile)
    if labels != expected:
        return CorrectnessEvaluation(
            "fail", entries, error=f"checksum sequence mismatch: expected {expected}, got {labels}"
        )
    missing = [index for index, entry in enumerate(entries) if entry.nonfinite is None]
    if missing:
        return CorrectnessEvaluation(
            "fail", entries, error=f"checksum lines missing nonfinite at indices {missing}"
        )
    nonfinite = sum(entry.nonfinite or 0 for entry in entries)
    if nonfinite:
        return CorrectnessEvaluation(
            "fail", entries, nonfinite_logits=nonfinite, error=f"non-finite logits: {nonfinite}"
        )
    if case.scheme not in VMM_SCHEMES:
        return CorrectnessEvaluation("finite", entries, nonfinite_logits=0)
    if reference is None:
        if case.scheme == ZERO_COPY:
            return CorrectnessEvaluation(
                "fail",
                entries,
                nonfinite_logits=0,
                error="zero-copy result has no same-profile, same-context DMA checksum reference",
            )
        return CorrectnessEvaluation("finite_baseline", entries, nonfinite_logits=0)
    reference_scheme, wanted = reference
    mismatch = base.compare_checksum_sequences(
        [entry.identity() for entry in wanted], [entry.identity() for entry in entries]
    )
    wanted_hash = checksum_sha256(wanted)
    if mismatch:
        return CorrectnessEvaluation(
            "fail", entries, reference_scheme, wanted_hash, 0, mismatch
        )
    return CorrectnessEvaluation("match", entries, reference_scheme, wanted_hash, 0)


def blank_row() -> dict[str, str]:
    return {field: "" for field in CSV_FIELDS}


def read_rows(path: Path) -> dict[tuple[str, str, str, int], dict[str, str]]:
    rows: dict[tuple[str, str, str, int], dict[str, str]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"scheme", "control_key", "profile", "context_tokens", "status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Q8 resume CSV has an incompatible header")
        for row in reader:
            key = (
                row["scheme"], row["control_key"], row["profile"], int(row["context_tokens"])
            )
            rows[key] = row
    return rows


def validate_resume_rows(
    rows: Iterable[dict[str, str]], manifest_hash: str, model: ModelInfo
) -> None:
    for row in rows:
        if row.get("manifest_sha256") != manifest_hash:
            raise ValueError("resume CSV was produced by a different manifest")
        if row.get("model_file_bytes") != str(model.file_bytes):
            raise ValueError("resume CSV model file size differs from stable Q8")
        if row.get("eligible_tensor_bytes") != str(model.eligible_tensor_bytes):
            raise ValueError("resume CSV eligible tensor byte count differs from stable Q8")
        if row.get("gguf_tensor_count") != str(model.tensor_count):
            raise ValueError("resume CSV tensor count differs from stable Q8")
        if row.get("model_context_length") != str(model.context_length):
            raise ValueError("resume CSV context limit differs from stable Q8")
        if row.get("model_metadata_sha256") != model.metadata_sha256:
            raise ValueError("resume CSV metadata identity differs from stable Q8")
        if row.get("model_sha256") != model.file_sha256:
            raise ValueError("resume CSV full model hash differs from stable Q8")


def parse_checksum_json(raw: str) -> tuple[base.LogitChecksum, ...] | None:
    try:
        values = json.loads(raw)
        return tuple(
            base.LogitChecksum(
                str(item["label"]), str(item["checksum"]).lower(), int(item["nonfinite"])
            )
            for item in values
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def is_completed(row: dict[str, str] | None, retry_errors: bool) -> bool:
    if row is None:
        return False
    return row.get("status") in TERMINAL_STATUSES if retry_errors else True


def control_sort_value(scheme: str, control: str) -> Decimal:
    value = Decimal(control.split(":", 1)[1])
    return -value if scheme == NATIVE else value


def shallow_origin(
    rows: Iterable[dict[str, str]], scheme: str, profile: str, minimum_context: int
) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if row.get("scheme") == scheme
        and row.get("profile") == profile
        and row.get("context_tokens") == str(minimum_context)
        and row.get("status") == "ok"
    ]
    if not candidates:
        return None
    flagged = [row for row in candidates if row.get("is_scheme_origin") == "1"]
    if len(flagged) > 1:
        raise ValueError(f"multiple immutable {scheme}/{profile} scheme origins in CSV")
    if flagged:
        return flagged[0]
    # Compatibility fallback for an interrupted row written before the origin
    # marker existed: append order is represented by the timestamp and never
    # by a later-discovered lower offload percentage.
    return min(candidates, key=lambda row: row.get("case_time_utc", ""))


def global_origin(
    rows: Iterable[dict[str, str]], profile: str, minimum_context: int
) -> dict[str, str] | None:
    values = [
        row
        for row in rows
        if row.get("profile") == profile
        and row.get("context_tokens") == str(minimum_context)
        and row.get("status") == "ok"
        and row.get("scheme") in {DIRECT, DMA}
    ]
    flagged = [row for row in values if row.get("is_global_origin") == "1"]
    if len(flagged) > 1:
        raise ValueError(f"multiple immutable {profile} global origins in CSV")
    if flagged:
        return flagged[0]
    if not values:
        return None
    return min(values, key=lambda row: row.get("case_time_utc", ""))


def row_for_result(
    manifest: dict[str, Any],
    manifest_hash: str,
    run_id: str,
    model: ModelInfo,
    case: Q8Case,
    env: dict[str, str],
    command: list[str],
    result: base.CommandResult,
    status: str,
    perf: ParsedPerformance | None,
    error: str,
    correctness: CorrectnessEvaluation,
    scheme_origin_row: dict[str, str] | None,
    global_origin_row: dict[str, str] | None,
) -> dict[str, str]:
    if status == "ok" and correctness.status == "fail":
        status = "correctness_fail"
        error = correctness.error
    profile = profile_for(manifest, case.profile)
    prompt = case.context_tokens - profile.generation_tokens
    layout = base.parse_hybrid_layout(result.stderr)
    native = base.parse_native_buffers(result.stderr)
    expected = (
        str(manifest["sweep"].get("direct_probe_expected", "oom"))
        if case.scheme == DIRECT and case.probe_kind == "shallow"
        else ""
    )
    row = blank_row()
    row.update(
        schema_version=str(SCHEMA_VERSION),
        run_id=run_id,
        case_time_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=manifest_hash,
        experiment_id=str(manifest.get("experiment_id", "")),
        scheme=case.scheme,
        phase=case.phase,
        profile=case.profile,
        probe_kind=case.probe_kind,
        control_key=case.control_key(),
        target_offload_pp=(
            f"{case.target_offload_pp:.9f}" if case.target_offload_pp is not None else ""
        ),
        page_budget=str(case.page_budget) if case.page_budget is not None else "",
        native_ngl=str(case.native_ngl) if case.native_ngl is not None else "",
        context_tokens=str(case.context_tokens),
        actual_context_tokens=str(perf.actual_context_tokens) if perf else "",
        reserved_context_tokens=str(perf.actual_context_tokens) if perf else str(case.context_tokens),
        filled_context_tokens=str(case.context_tokens) if status == "ok" else "",
        prompt_tokens=str(prompt),
        generation_tokens=str(profile.generation_tokens),
        repetitions=str(profile.repetitions),
        total_prompt_eval_tokens=str(perf.prompt_samples) if perf else "",
        total_decode_eval_tokens=str(perf.decode_samples) if perf else "",
        populated_context_proven="1" if status == "ok" else "0",
        status=status,
        exit_code=str(result.returncode),
        elapsed_s=f"{result.elapsed_s:.3f}",
        expected_outcome=expected,
        expectation_met=("1" if status == expected else "0") if expected else "",
        model_file_bytes=str(model.file_bytes),
        eligible_tensor_bytes=str(model.eligible_tensor_bytes),
        gguf_tensor_count=str(model.tensor_count),
        model_context_length=str(model.context_length),
        model_metadata_sha256=model.metadata_sha256,
        model_sha256=model.file_sha256,
        offload_budget_basis="eligible_tensor_bytes",
        command_json=json.dumps(
            {"env": env, "argv": command}, separators=(",", ":"), sort_keys=True
        ),
        layout_json=json.dumps(layout, separators=(",", ":"), sort_keys=True) if layout else "",
        remote_artifacts_created="0",
        error_tail=error,
    )
    row.update(base.summarize_telemetry(result.telemetry))
    if perf:
        row.update(
            perf_prompt_ms=f"{perf.prompt_ms:.6f}",
            perf_decode_ms=f"{perf.decode_ms:.6f}",
            prefill_tok_s=f"{perf.prefill_tok_s:.6f}",
            decode_tok_s=f"{perf.decode_tok_s:.6f}",
        )
    for key, value in native.items():
        row[key] = f"{value:.3f}"
    for key in (
        "requested_pages", "gross_pages", "gross_remote_bytes", "staging_bytes",
        "net_saved_bytes", "selected_tensors", "selected_layers",
    ):
        if key in layout:
            row[key] = str(layout[key])
    gross = int(row["gross_remote_bytes"] or 0)
    staging = int(row["staging_bytes"] or 0)
    net = int(row["net_saved_bytes"] or 0)
    if gross:
        row["gross_remote_mib"] = f"{gross / MIB:.3f}"
        row["gross_remote_pp"] = f"{100 * gross / model.eligible_tensor_bytes:.6f}"
    if staging:
        row["staging_mib"] = f"{staging / MIB:.3f}"
    if net:
        row["net_saved_mib"] = f"{net / MIB:.3f}"
        row["net_saved_pp"] = f"{100 * net / model.eligible_tensor_bytes:.6f}"

    row["checksum_status"] = correctness.status
    row["checksum_reference_scheme"] = correctness.reference_scheme
    row["checksum_reference_sha256"] = correctness.reference_sha256
    if correctness.nonfinite_logits is not None:
        row["nonfinite_logits"] = str(correctness.nonfinite_logits)
    if correctness.entries:
        row["checksum_sequence_json"] = json.dumps(
            [
                {"label": entry.label, "checksum": entry.checksum, "nonfinite": entry.nonfinite}
                for entry in correctness.entries
            ],
            separators=(",", ":"),
        )

    if status == "ok" and scheme_origin_row is None and case.scheme != DIRECT:
        scheme_origin_row = row
        row["is_scheme_origin"] = "1"
    if status == "ok" and global_origin_row is None and case.scheme in {DIRECT, DMA}:
        global_origin_row = row
        row["is_global_origin"] = "1"

    def retention(origin: dict[str, str], prefix: str) -> None:
        row[f"{prefix}_origin_scheme"] = origin["scheme"]
        row[f"{prefix}_origin_control"] = origin["control_key"]
        row[f"{prefix}_origin_context_tokens"] = origin["context_tokens"]
        row[f"prefill_retention_vs_{prefix}_origin"] = (
            f"{float(row['prefill_tok_s']) / float(origin['prefill_tok_s']):.6f}"
        )
        row[f"decode_retention_vs_{prefix}_origin"] = (
            f"{float(row['decode_tok_s']) / float(origin['decode_tok_s']):.6f}"
        )

    if status == "ok" and scheme_origin_row is not None and case.scheme != DIRECT:
        retention(scheme_origin_row, "scheme")
    if status == "ok" and global_origin_row is not None:
        retention(global_origin_row, "global")
    return row


def retention_is_low(row: dict[str, str], threshold: float) -> bool:
    if row.get("status") != "ok":
        return False
    try:
        return (
            float(row["prefill_retention_vs_global_origin"]) <= threshold
            or float(row["decode_retention_vs_global_origin"]) <= threshold
        )
    except (KeyError, ValueError):
        return False


def control_values_from_row(
    row: dict[str, str]
) -> tuple[Decimal | None, int | None, int | None]:
    if row["scheme"] == NATIVE:
        return None, None, int(row["native_ngl"])
    return (
        Decimal(row.get("target_offload_pp") or "0"),
        int(row.get("page_budget") or 0) if row["scheme"] in VMM_SCHEMES else None,
        None,
    )


class Q8SweepRunner:
    def __init__(
        self,
        manifest: dict[str, Any],
        manifest_hash: str,
        output: Path,
        frontier_output: Path,
        model: ModelInfo,
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
        self.frontier_output = frontier_output
        self.model = model
        self.dry_run = dry_run
        self.retry_errors = retry_errors
        self.show_telemetry = show_telemetry
        self.max_cases = max_cases
        self.profile = base.profiling_config(manifest, profile_mode)
        if profile_raw:
            self.profile["raw_sidecar"] = True
        self.profile_output = (
            profile_output
            if profile_output is not None
            else output.with_name(f"{output.stem}.profile.jsonl")
        )
        self.executed = 0
        self.run_id = str(uuid.uuid4())
        # Dry-run uses planning byte counts before the remote GGUF exists, so
        # it must not attempt to validate a real result file against those
        # placeholders.
        self.rows = {} if dry_run else read_rows(output)
        validate_resume_rows(self.rows.values(), manifest_hash, model)
        self.planned_keys = set(self.rows)
        self.output_stream: TextIO | None = None
        self.writer: csv.DictWriter | None = None
        self.references: dict[tuple[str, int], tuple[str, tuple[base.LogitChecksum, ...]]] = {}
        self._load_references()

    @property
    def minimum_context(self) -> int:
        return int(self.manifest["context"].get("minimum_context_tokens", 512))

    def _load_references(self) -> None:
        for row in sorted(self.rows.values(), key=lambda value: value.get("case_time_utc", "")):
            if row.get("scheme") != DMA or row.get("status") != "ok":
                continue
            entries = parse_checksum_json(row.get("checksum_sequence_json", ""))
            if entries is not None:
                self.references.setdefault(
                    (row["profile"], int(row["context_tokens"])), (DMA, entries)
                )

    def __enter__(self) -> "Q8SweepRunner":
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
        if not self.dry_run:
            write_frontier_csv(self.frontier_output, self.rows.values())

    def reached_limit(self) -> bool:
        return self.max_cases is not None and self.executed >= self.max_cases

    def origins(self, scheme: str, profile: str) -> tuple[dict[str, str] | None, dict[str, str] | None]:
        values = list(self.rows.values())
        return (
            shallow_origin(values, scheme, profile, self.minimum_context),
            global_origin(values, profile, self.minimum_context),
        )

    def run_case(self, case: Q8Case) -> dict[str, str] | None:
        existing = self.rows.get(case.key())
        if is_completed(existing, self.retry_errors):
            print(
                f"skip {case.scheme} {case.control_key()} profile={case.profile} "
                f"ctx={case.context_tokens} status={existing.get('status')}",
                flush=True,
            )
            return existing
        if self.dry_run and case.key() in self.planned_keys:
            return None
        if self.reached_limit():
            return None
        if (
            case.scheme == ZERO_COPY
            and not self.dry_run
            and (case.profile, case.context_tokens) not in self.references
        ):
            if not self.ensure_dma_reference(case):
                return None
        env, command = command_for_case(self.manifest, case)
        base.enable_backend_profile(env, self.profile)
        profile = profile_for(self.manifest, case.profile)
        prompt = case.context_tokens - profile.generation_tokens
        print(
            f"case scheme={case.scheme} phase={case.phase} kind={case.probe_kind} "
            f"control={case.control_key()} profile={case.profile} C={case.context_tokens} "
            f"P={prompt} G={profile.generation_tokens} R={profile.repetitions}",
            flush=True,
        )
        displayed = [
            "env", "-u", "LD_PRELOAD",
            *[f"{key}={value}" for key, value in sorted(env.items())],
            *command,
        ]
        print(f"remote command: {base.shell_join(displayed)}", flush=True)
        self.executed += 1
        if self.dry_run:
            self.planned_keys.add(case.key())
            return None

        wrapper = base.remote_wrapper(
            env,
            command,
            float(self.profile["interval_s"]),
            str(self.manifest["remote"].get("gpu_id", 0)),
            self.profile,
        )
        result = base.execute_wrapper(
            self.manifest,
            wrapper,
            timeout_s=float(self.manifest["sweep"].get("case_timeout_s", 7200)),
            show_telemetry=self.show_telemetry,
        )
        status, perf, error = classify_result(self.manifest, case, result, self.model)
        reference = self.references.get((case.profile, case.context_tokens))
        correctness = evaluate_correctness(
            self.manifest, case, result, status, reference
        )
        scheme_origin_row, global_origin_row = self.origins(case.scheme, case.profile)
        row = row_for_result(
            self.manifest,
            self.manifest_hash,
            self.run_id,
            self.model,
            case,
            env,
            command,
            result,
            status,
            perf,
            error,
            correctness,
            scheme_origin_row,
            global_origin_row,
        )
        assert self.writer is not None and self.output_stream is not None
        self.writer.writerow(row)
        self.output_stream.flush()
        os.fsync(self.output_stream.fileno())
        self.rows[case.key()] = row
        if self.profile["enabled"]:
            base.append_profile_sidecar(
                self.profile_output,
                base.profile_sidecar_record(
                    profile=self.profile,
                    run_id=self.run_id,
                    manifest_sha256=self.manifest_hash,
                    case_key={
                        "scheme": case.scheme,
                        "phase": case.phase,
                        "profile": case.profile,
                        "probe_kind": case.probe_kind,
                        "control": case.control_key(),
                        "context_tokens": case.context_tokens,
                        "target_offload_pp": (
                            canonical_decimal(case.target_offload_pp)
                            if case.target_offload_pp is not None
                            else None
                        ),
                        "page_budget": case.page_budget,
                        "native_ngl": case.native_ngl,
                    },
                    status=row["status"],
                    result=result,
                    command_identity={"env": env, "argv": command},
                ),
            )
        if row["status"] == "ok" and case.scheme == DMA and reference is None:
            entries = parse_checksum_json(row["checksum_sequence_json"])
            if entries is not None:
                self.references[(case.profile, case.context_tokens)] = (DMA, entries)
        write_frontier_csv(self.frontier_output, self.rows.values())
        print(
            f"result status={row['status']} pp={row['prefill_tok_s'] or '-'} "
            f"tg={row['decode_tok_s'] or '-'} filled={row['filled_context_tokens'] or '-'} "
            f"peak={row['gpu_process_peak_mib'] or '-'}MiB",
            flush=True,
        )
        if row["status"] == "correctness_fail":
            raise RuntimeError(
                f"correctness failure {case.scheme} {case.control_key()} C={case.context_tokens}: "
                f"{row['error_tail']}"
            )
        if row["status"] == "transport_error":
            raise RuntimeError(f"SSH transport failure: {row['error_tail']}")
        if row["status"] in {"error", "parse_error", "timeout"}:
            raise RuntimeError(
                f"benchmark infrastructure failure {case.scheme} {case.control_key()}: "
                f"{row['error_tail']}"
            )
        return row

    def ensure_dma_reference(self, zero_case: Q8Case) -> bool:
        """Materialize an exact-profile/context DMA checksum before zero-copy."""

        key = (zero_case.profile, zero_case.context_tokens)
        if key in self.references:
            return True
        seed = max(0, int(zero_case.page_budget or 0))
        row = self.find_min_vmm_pages(
            DMA,
            PHASE_REFERENCE,
            zero_case.profile,
            zero_case.context_tokens,
            seed_pages=seed,
            probe_kind="checksum_reference",
        )
        if row is None or key not in self.references:
            return False
        return True

    def case_for_control(
        self,
        scheme: str,
        phase: str,
        profile: str,
        probe_kind: str,
        context_tokens: int,
        *,
        target_pp: Decimal | None = None,
        page_budget: int | None = None,
        native_ngl: int | None = None,
    ) -> Q8Case:
        return build_case(
            self.manifest,
            self.model,
            scheme,
            phase,
            profile,
            probe_kind,
            context_tokens,
            target_pp=target_pp,
            page_budget=page_budget,
            native_ngl=native_ngl,
        )

    def rows_for_control(
        self, scheme: str, control: str, profile: str
    ) -> list[dict[str, str]]:
        return [
            row
            for (row_scheme, row_control, row_profile, _), row in self.rows.items()
            if row_scheme == scheme and row_control == control and row_profile == profile
        ]

    def probe_capacity(
        self,
        scheme: str,
        phase: str,
        *,
        profile: str,
        step_tokens: int,
        seed_context: int,
        target_pp: Decimal | None = None,
        native_ngl: int | None = None,
    ) -> dict[str, str] | None:
        control = (
            f"ngl:{native_ngl}"
            if scheme == NATIVE
            else (
                f"pages:{page_budget_for_pp(self.manifest, self.model, target_pp or Decimal(0))}"
                if scheme in VMM_SCHEMES
                else "local"
            )
        )
        shallow = self.run_case(
            self.case_for_control(
                scheme,
                phase,
                profile,
                "shallow",
                self.minimum_context,
                target_pp=target_pp,
                native_ngl=native_ngl,
            )
        )
        if self.reached_limit() or shallow is None:
            return None
        if shallow.get("status") != "ok":
            return shallow

        prior = self.rows_for_control(scheme, control, profile)
        successes = [row for row in prior if row.get("status") == "ok"]
        best = max(successes, key=lambda row: int(row["context_tokens"]))
        maximum = int(self.manifest["context"].get("maximum_context_tokens", 262144))
        quantum = int(self.manifest["context"].get("quantum_tokens", 256))
        seed = max(self.minimum_context, min(maximum, seed_context - seed_context % quantum))
        if seed > int(best["context_tokens"]):
            seeded = self.run_case(
                self.case_for_control(
                    scheme,
                    phase,
                    profile,
                    "frontier",
                    seed,
                    target_pp=target_pp,
                    native_ngl=native_ngl,
                )
            )
            if self.reached_limit() or seeded is None:
                return best
            if seeded.get("status") == "ok":
                best = seeded

        while not self.reached_limit():
            candidate = int(best["context_tokens"]) + step_tokens
            if candidate > maximum:
                return best
            known = self.rows.get((scheme, control, profile, candidate))
            if known is not None and known.get("status") == "oom":
                return best
            row = self.run_case(
                self.case_for_control(
                    scheme,
                    phase,
                    profile,
                    "frontier",
                    candidate,
                    target_pp=target_pp,
                    native_ngl=native_ngl,
                )
            )
            if row is None:
                return best
            if row.get("status") == "ok":
                best = row
                continue
            if row.get("status") in {"oom", "cuda_capacity_fail", "unsupported"}:
                return best
        return best

    def run_direct(self) -> None:
        # Usually Q8 fails at the shallow load gate.  If it unexpectedly fits,
        # continue with the same populated-context frontier search instead of
        # silently treating Cmin as the all-local capacity.
        frontier = self.probe_capacity(
            DIRECT,
            PHASE_DIRECT,
            profile=SCREEN_PROFILE,
            step_tokens=int(self.manifest["context"].get("coarse_step_tokens", 4096)),
            seed_context=self.minimum_context,
            target_pp=Decimal(0),
        )
        if frontier is None or frontier.get("status") != "ok" or self.reached_limit():
            return
        self.probe_capacity(
            DIRECT,
            PHASE_DIRECT,
            profile=SCREEN_PROFILE,
            step_tokens=int(self.manifest["context"].get("fine_step_tokens", 256)),
            seed_context=int(frontier["context_tokens"]),
            target_pp=Decimal(0),
        )

    def direct_frontier_context(self) -> int:
        candidates = [
            int(row["context_tokens"])
            for row in self.rows.values()
            if row.get("scheme") == DIRECT
            and row.get("profile") == SCREEN_PROFILE
            and row.get("status") == "ok"
        ]
        return max(candidates, default=self.minimum_context)

    def ensure_profile_origin(self, scheme: str, profile: str) -> None:
        values = list(self.rows.values())
        global_screen = global_origin(values, SCREEN_PROFILE, self.minimum_context)
        if global_screen is None:
            raise RuntimeError("cannot confirm threshold without an operational staging origin")
        global_profile = global_origin(values, profile, self.minimum_context)
        if global_profile is None:
            target_pp, page_budget, native_ngl = control_values_from_row(global_screen)
            self.run_case(
                self.case_for_control(
                    global_screen["scheme"],
                    PHASE_CONFIRM,
                    profile,
                    "origin",
                    self.minimum_context,
                    target_pp=target_pp,
                    page_budget=page_budget,
                    native_ngl=native_ngl,
                )
            )
        if scheme == DIRECT:
            return
        scheme_profile = shallow_origin(
            self.rows.values(), scheme, profile, self.minimum_context
        )
        if scheme_profile is None:
            scheme_screen = shallow_origin(
                self.rows.values(), scheme, SCREEN_PROFILE, self.minimum_context
            )
            if scheme_screen is None:
                raise RuntimeError(f"cannot confirm threshold without {scheme} origin")
            target_pp, page_budget, native_ngl = control_values_from_row(scheme_screen)
            self.run_case(
                self.case_for_control(
                    scheme,
                    PHASE_CONFIRM,
                    profile,
                    "origin",
                    self.minimum_context,
                    target_pp=target_pp,
                    page_budget=page_budget,
                    native_ngl=native_ngl,
                )
            )

    def confirm_frontier(self, row: dict[str, str]) -> dict[str, str] | None:
        scheme = row["scheme"]
        self.ensure_profile_origin(scheme, CONFIRM_PROFILE)
        target_pp, page_budget, native_ngl = control_values_from_row(row)
        return self.run_case(
            self.case_for_control(
                scheme,
                PHASE_CONFIRM,
                CONFIRM_PROFILE,
                "frontier_confirmation",
                int(row["context_tokens"]),
                target_pp=target_pp,
                page_budget=page_budget,
                native_ngl=native_ngl,
            )
        )

    def run_vmm_coarse(self, scheme: str, phase: str) -> None:
        sweep = self.manifest["sweep"]
        start = Decimal(str(sweep.get("coarse_start_pp", "0")))
        maximum = Decimal(str(sweep.get("max_offload_pp", "95")))
        pp_step = Decimal(str(sweep.get("coarse_step_pp", "1")))
        ctx_step = int(self.manifest["context"].get("coarse_step_tokens", 4096))
        threshold = float(sweep.get("stop_retention", 0.10))
        consecutive_needed = int(sweep.get("stop_consecutive", 1))
        consecutive = 0
        seed = self.direct_frontier_context()
        for target in decimal_range(start, maximum, pp_step):
            frontier = self.probe_capacity(
                scheme,
                phase,
                profile=SCREEN_PROFILE,
                step_tokens=ctx_step,
                seed_context=seed,
                target_pp=target,
            )
            if self.reached_limit():
                return
            if frontier is None or frontier.get("status") != "ok":
                if frontier and frontier.get("status") == "unsupported":
                    return
                continue
            seed = int(frontier["context_tokens"])
            if retention_is_low(frontier, threshold):
                confirmed = self.confirm_frontier(frontier)
                if confirmed is not None and retention_is_low(confirmed, threshold):
                    consecutive += 1
                    if consecutive >= consecutive_needed:
                        print(
                            f"stop coarse {scheme}: confirmed <= {threshold:.1%} at "
                            f"{target}% and C={seed}",
                            flush=True,
                        )
                        return
                else:
                    consecutive = 0
            else:
                consecutive = 0

    def run_native_coarse(self) -> None:
        sweep = self.manifest["sweep"]
        start = int(sweep.get("native_ngl_start", self.manifest["model"].get("gpu_layers", 61)))
        stop = int(sweep.get("native_ngl_stop", 0))
        ctx_step = int(self.manifest["context"].get("coarse_step_tokens", 4096))
        threshold = float(sweep.get("stop_retention", 0.10))
        consecutive_needed = int(sweep.get("stop_consecutive", 1))
        consecutive = 0
        seed = self.direct_frontier_context()
        for ngl in range(start, stop - 1, -1):
            frontier = self.probe_capacity(
                NATIVE,
                PHASE_ALTERNATES,
                profile=SCREEN_PROFILE,
                step_tokens=ctx_step,
                seed_context=seed,
                native_ngl=ngl,
            )
            if self.reached_limit():
                return
            if frontier is None or frontier.get("status") != "ok":
                continue
            seed = int(frontier["context_tokens"])
            if retention_is_low(frontier, threshold):
                confirmed = self.confirm_frontier(frontier)
                if confirmed is not None and retention_is_low(confirmed, threshold):
                    consecutive += 1
                    if consecutive >= consecutive_needed:
                        return
                else:
                    consecutive = 0
            else:
                consecutive = 0

    def coarse_bounds(self, scheme: str) -> tuple[Decimal, Decimal] | None:
        successful: dict[Decimal, dict[str, str]] = {}
        for row in self.rows.values():
            if (
                row.get("scheme") != scheme
                or row.get("profile") != SCREEN_PROFILE
                or row.get("status") != "ok"
                or not row.get("target_offload_pp")
                or row.get("phase") not in {PHASE_STAGING_COARSE, PHASE_ALTERNATES, PHASE_FINE}
            ):
                continue
            target = Decimal(row["target_offload_pp"])
            current = successful.get(target)
            if current is None or int(row["context_tokens"]) > int(current["context_tokens"]):
                successful[target] = row
        if not successful:
            return None
        first_fit = min(successful)
        threshold = float(self.manifest["sweep"].get("stop_retention", 0.10))
        crossing = next(
            (target for target in sorted(successful) if retention_is_low(successful[target], threshold)),
            max(successful),
        )
        coarse_step = Decimal(str(self.manifest["sweep"].get("coarse_step_pp", "1")))
        return max(Decimal(0), first_fit - coarse_step), crossing

    def run_vmm_fine(self, scheme: str) -> None:
        bounds = self.coarse_bounds(scheme)
        if bounds is None:
            print(f"fine {scheme}: no successful coarse point", flush=True)
            return
        start, stop = bounds
        pp_step = Decimal(str(self.manifest["sweep"].get("dense_step_pp", "0.1")))
        ctx_step = int(self.manifest["context"].get("fine_step_tokens", 256))
        threshold = float(self.manifest["sweep"].get("stop_retention", 0.10))
        seed = self.direct_frontier_context()
        consecutive_needed = int(self.manifest["sweep"].get("stop_consecutive", 1))
        consecutive = 0
        for target in decimal_range(start, stop, pp_step):
            frontier = self.probe_capacity(
                scheme,
                PHASE_FINE,
                profile=SCREEN_PROFILE,
                step_tokens=ctx_step,
                seed_context=seed,
                target_pp=target,
            )
            if self.reached_limit():
                return
            if frontier is None or frontier.get("status") != "ok":
                if frontier and frontier.get("status") == "unsupported":
                    return
                continue
            seed = int(frontier["context_tokens"])
            if retention_is_low(frontier, threshold):
                confirmed = self.confirm_frontier(frontier)
                if confirmed is not None and retention_is_low(confirmed, threshold):
                    consecutive += 1
                    if consecutive >= consecutive_needed:
                        print(
                            f"stop fine {scheme}: {consecutive_needed} confirmed points "
                            f"<= {threshold:.1%}, ending at {target}% and C={seed}",
                            flush=True,
                        )
                        return
                else:
                    consecutive = 0
            else:
                consecutive = 0

    @staticmethod
    def is_capacity_failure(row: dict[str, str] | None) -> bool:
        return row is not None and row.get("status") in {
            "oom", "cuda_capacity_fail"
        }

    def maximum_vmm_pages(self) -> int:
        page_bytes = int(self.manifest["model"]["page_bytes"])
        raw_pages = math.ceil(self.model.eligible_tensor_bytes / page_bytes)
        # Per-tensor VMM mappings round independently.  Adding tensor_count is
        # a conservative upper bound on all such partial-page padding; the
        # exact minimum found by bisection is unaffected by this loose ceiling.
        return raw_pages + self.model.tensor_count

    def vmm_case(
        self,
        scheme: str,
        phase: str,
        profile: str,
        probe_kind: str,
        context_tokens: int,
        page_budget: int,
    ) -> Q8Case:
        return build_case(
            self.manifest,
            self.model,
            scheme,
            phase,
            profile,
            probe_kind,
            context_tokens,
            page_budget=page_budget,
        )

    def find_min_vmm_pages(
        self,
        scheme: str,
        phase: str,
        profile: str,
        context_tokens: int,
        *,
        seed_pages: int,
        probe_kind: str = "minimum_page_search",
    ) -> dict[str, str] | None:
        """Bracket and bisect to the exact minimum successful 2 MiB page budget."""

        if scheme not in VMM_SCHEMES:
            raise ValueError("page search requires a VMM scheme")
        maximum = self.maximum_vmm_pages()
        seed = min(maximum, max(0, int(seed_pages)))

        def existing(pages: int) -> dict[str, str] | None:
            return self.rows.get((scheme, f"pages:{pages}", profile, context_tokens))

        def probe(pages: int) -> dict[str, str] | None:
            return self.run_case(
                self.vmm_case(
                    scheme, phase, profile, probe_kind, context_tokens, pages
                )
            )

        successes = [
            row
            for row in self.rows.values()
            if row.get("scheme") == scheme
            and row.get("profile") == profile
            and row.get("context_tokens") == str(context_tokens)
            and row.get("status") == "ok"
            and row.get("page_budget")
        ]
        if successes:
            best = min(successes, key=lambda row: int(row["page_budget"]))
            pages = int(best["page_budget"])
            if pages == 0 or self.is_capacity_failure(existing(pages - 1)):
                return best

        if self.dry_run:
            probe(seed)
            return None

        zero = probe(0)
        if zero is None or self.reached_limit():
            return zero if zero and zero.get("status") == "ok" else None
        if zero.get("status") == "ok":
            return zero
        if zero.get("status") == "unsupported":
            return None
        if not self.is_capacity_failure(zero):
            raise RuntimeError(f"unexpected VMM page-search status at 0 pages: {zero.get('status')}")

        low = 0
        high: int | None = None
        high_row: dict[str, str] | None = None
        candidate = max(1, seed)
        bracket_step = max(
            1,
            page_budget_for_pp(
                self.manifest,
                self.model,
                Decimal(str(self.manifest["sweep"].get("page_bracket_step_pp", "1"))),
            ),
        )
        while candidate <= maximum:
            row = probe(candidate)
            if row is None or self.reached_limit():
                return None
            if row.get("status") == "ok":
                high, high_row = candidate, row
                break
            if row.get("status") == "unsupported":
                return None
            if not self.is_capacity_failure(row):
                raise RuntimeError(
                    f"unexpected VMM page-search status at {candidate} pages: {row.get('status')}"
                )
            low = candidate
            distance = max(bracket_step, candidate - seed, 1)
            candidate = min(maximum, candidate + distance)
            if candidate == low:
                break
        if high is None:
            if low != maximum:
                row = probe(maximum)
                if row is not None and row.get("status") == "ok":
                    high, high_row = maximum, row
            if high is None:
                return None

        while high - low > 1 and not self.reached_limit():
            middle = (low + high) // 2
            row = probe(middle)
            if row is None:
                return None
            if row.get("status") == "ok":
                high, high_row = middle, row
            elif self.is_capacity_failure(row):
                low = middle
            elif row.get("status") == "unsupported":
                return None
            else:
                raise RuntimeError(
                    f"unexpected VMM bisection status at {middle} pages: {row.get('status')}"
                )
        return high_row

    def confirm_minimum_row(self, row: dict[str, str]) -> dict[str, str] | None:
        scheme = row["scheme"]
        context = int(row["context_tokens"])
        self.ensure_profile_origin(scheme, CONFIRM_PROFILE)
        if scheme in VMM_SCHEMES:
            pages = int(row["page_budget"])
            return self.run_case(
                self.vmm_case(
                    scheme,
                    PHASE_CONFIRM,
                    CONFIRM_PROFILE,
                    "frontier_confirmation",
                    context,
                    pages,
                )
            )
        if scheme == NATIVE:
            return self.run_case(
                build_case(
                    self.manifest,
                    self.model,
                    NATIVE,
                    PHASE_CONFIRM,
                    CONFIRM_PROFILE,
                    "frontier_confirmation",
                    context,
                    native_ngl=int(row["native_ngl"]),
                )
            )
        return None

    def run_vmm_context_curve(
        self,
        scheme: str,
        phase: str,
        *,
        context_step: int,
        stop_context: int | None = None,
    ) -> None:
        maximum = min(
            int(self.manifest["context"].get("maximum_context_tokens", 262144)),
            stop_context or self.model.context_length,
        )
        threshold = float(self.manifest["sweep"].get("stop_retention", 0.10))
        consecutive_needed = int(self.manifest["sweep"].get("stop_consecutive", 2))
        consecutive = 0
        previous_pages = 0

        contexts: list[int] = [self.minimum_context]
        start = self.direct_frontier_context() + context_step
        contexts.extend(range(start, maximum + 1, context_step))
        for context in sorted(set(contexts)):
            if self.reached_limit():
                return
            row = self.find_min_vmm_pages(
                scheme,
                phase,
                SCREEN_PROFILE,
                context,
                seed_pages=previous_pages,
            )
            if row is None:
                continue
            previous_pages = int(row["page_budget"])
            if retention_is_low(row, threshold):
                confirmed = self.confirm_minimum_row(row)
                if confirmed is not None and retention_is_low(confirmed, threshold):
                    consecutive += 1
                    if consecutive >= consecutive_needed:
                        return
                else:
                    consecutive = 0
            else:
                consecutive = 0

    def find_max_native_ngl(
        self, phase: str, profile: str, context_tokens: int
    ) -> dict[str, str] | None:
        start = int(self.manifest["sweep"].get("native_ngl_start", 61))
        stop = int(self.manifest["sweep"].get("native_ngl_stop", 0))

        def probe(ngl: int) -> dict[str, str] | None:
            return self.run_case(
                build_case(
                    self.manifest,
                    self.model,
                    NATIVE,
                    phase,
                    profile,
                    "minimum_layer_search",
                    context_tokens,
                    native_ngl=ngl,
                )
            )

        top = probe(start)
        if top is None or self.reached_limit():
            return top if top and top.get("status") == "ok" else None
        if top.get("status") == "ok":
            return top
        bottom = probe(stop)
        if bottom is None or bottom.get("status") != "ok":
            return None
        low_fit, high_fail = stop, start
        best = bottom
        while high_fail - low_fit > 1 and not self.reached_limit():
            middle = (low_fit + high_fail) // 2
            row = probe(middle)
            if row is None:
                return None
            if row.get("status") == "ok":
                low_fit, best = middle, row
            elif self.is_capacity_failure(row):
                high_fail = middle
            else:
                return None
        return best

    def run_native_context_curve(self) -> None:
        maximum = int(self.manifest["context"].get("maximum_context_tokens", 262144))
        step = int(self.manifest["context"].get("coarse_step_tokens", 4096))
        threshold = float(self.manifest["sweep"].get("stop_retention", 0.10))
        consecutive_needed = int(self.manifest["sweep"].get("stop_consecutive", 2))
        consecutive = 0
        contexts = [self.minimum_context]
        contexts.extend(range(self.direct_frontier_context() + step, maximum + 1, step))
        for context in sorted(set(contexts)):
            row = self.find_max_native_ngl(PHASE_ALTERNATES, SCREEN_PROFILE, context)
            if self.reached_limit():
                return
            if row is None:
                continue
            if retention_is_low(row, threshold):
                confirmed = self.confirm_minimum_row(row)
                if confirmed is not None and retention_is_low(confirmed, threshold):
                    consecutive += 1
                    if consecutive >= consecutive_needed:
                        return
                else:
                    consecutive = 0
            else:
                consecutive = 0

    def coarse_context_stop(self, scheme: str) -> int | None:
        contexts = [
            int(row["context_tokens"])
            for row in self.rows.values()
            if row.get("scheme") == scheme
            and row.get("phase") in {PHASE_STAGING_COARSE, PHASE_ALTERNATES}
            and row.get("profile") == SCREEN_PROFILE
            and row.get("status") == "ok"
        ]
        return max(contexts, default=None)

    def run(self) -> None:
        phases = self.manifest["sweep"].get("phases", list(PHASES))
        if PHASE_DIRECT in phases and not self.reached_limit():
            self.run_direct()
        if PHASE_STAGING_COARSE in phases and not self.reached_limit():
            self.run_vmm_context_curve(
                DMA,
                PHASE_STAGING_COARSE,
                context_step=int(self.manifest["context"].get("coarse_step_tokens", 4096)),
            )
        if PHASE_ALTERNATES in phases and not self.reached_limit():
            self.run_vmm_context_curve(
                ZERO_COPY,
                PHASE_ALTERNATES,
                context_step=int(self.manifest["context"].get("coarse_step_tokens", 4096)),
            )
            if not self.reached_limit():
                self.run_native_context_curve()
        if PHASE_FINE in phases:
            for scheme in self.manifest["sweep"].get("fine_schemes", list(VMM_SCHEMES)):
                if self.reached_limit():
                    return
                stop = self.coarse_context_stop(scheme)
                if stop is not None:
                    self.run_vmm_context_curve(
                        scheme,
                        PHASE_FINE,
                        context_step=int(self.manifest["context"].get("fine_step_tokens", 256)),
                        stop_context=stop,
                    )


def write_frontier_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    values = list(rows)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in values:
        if row.get("profile") != SCREEN_PROFILE:
            continue
        grouped.setdefault(
            (row.get("scheme", ""), int(row.get("context_tokens") or 0)), []
        ).append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRONTIER_FIELDS)
        writer.writeheader()
        for (scheme, context), candidates in sorted(
            grouped.items(),
            key=lambda item: (
                SCHEMES.index(item[0][0]) if item[0][0] in SCHEMES else len(SCHEMES),
                item[0][1],
            ),
        ):
            successes = [row for row in candidates if row.get("status") == "ok"]
            if not successes:
                continue
            if scheme in VMM_SCHEMES:
                frontier = min(successes, key=lambda row: int(row["page_budget"]))
            elif scheme == NATIVE:
                frontier = max(successes, key=lambda row: int(row["native_ngl"]))
            else:
                frontier = successes[0]
            control = frontier["control_key"]
            failed_pages = sorted(
                int(row["page_budget"])
                for row in candidates
                if scheme in VMM_SCHEMES
                and row.get("status") in {"oom", "cuda_capacity_fail"}
                and row.get("page_budget")
                and int(row["page_budget"]) < int(frontier["page_budget"])
            )
            failed_contexts = sorted(
                int(row["context_tokens"])
                for row in values
                if scheme == DIRECT
                and row.get("scheme") == scheme
                and row.get("profile") == SCREEN_PROFILE
                and row.get("status") in {"oom", "cuda_capacity_fail"}
                and int(row.get("context_tokens") or 0) > context
            )
            confirmations = [
                row
                for row in values
                if row.get("scheme") == scheme
                and row.get("control_key") == control
                and row.get("profile") == CONFIRM_PROFILE
                and row.get("context_tokens") == frontier["context_tokens"]
                and row.get("status") == "ok"
            ]
            confirmation = confirmations[-1] if confirmations else None
            output = {field: "" for field in FRONTIER_FIELDS}
            output.update(
                schema_version=str(SCHEMA_VERSION),
                scheme=scheme,
                control_key=control,
                target_offload_pp=frontier.get("target_offload_pp", ""),
                native_ngl=frontier.get("native_ngl", ""),
                frontier_phase=frontier.get("phase", ""),
                frontier_context_tokens=frontier["context_tokens"],
                actual_context_tokens=frontier.get("actual_context_tokens", ""),
                minimum_page_budget=frontier.get("page_budget", ""),
                last_failed_page_budget=str(failed_pages[-1]) if failed_pages else "",
                first_failed_context_tokens=str(failed_contexts[0]) if failed_contexts else "",
            )
            for field in (
                "prompt_tokens", "generation_tokens", "prefill_tok_s", "decode_tok_s",
                "prefill_retention_vs_scheme_origin", "decode_retention_vs_scheme_origin",
                "prefill_retention_vs_global_origin", "decode_retention_vs_global_origin",
                "gross_remote_mib", "net_saved_mib", "gpu_process_peak_mib",
                "gpu_device_peak_mib", "checksum_status",
            ):
                output[field] = frontier.get(field, "")
            if confirmation:
                output.update(
                    confirmed_prefill_tok_s=confirmation.get("prefill_tok_s", ""),
                    confirmed_decode_tok_s=confirmation.get("decode_tok_s", ""),
                    confirmed_prefill_retention_vs_global_origin=confirmation.get(
                        "prefill_retention_vs_global_origin", ""
                    ),
                    confirmed_decode_retention_vs_global_origin=confirmation.get(
                        "decode_retention_vs_global_origin", ""
                    ),
                )
            writer.writerow(output)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def plan_summary(manifest: dict[str, Any], model: ModelInfo) -> dict[str, Any]:
    sweep = manifest["sweep"]
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_order": [
            "wait_for_stable_download_and_parse_gguf_metadata",
            PHASE_DIRECT,
            PHASE_STAGING_COARSE,
            f"{PHASE_ALTERNATES}:cuda_zero_copy",
            f"{PHASE_ALTERNATES}:native_cpu_layers",
            PHASE_FINE,
        ],
        "model_file_bytes": model.file_bytes,
        "eligible_tensor_bytes": model.eligible_tensor_bytes,
        "model_context_length": model.context_length,
        "model_metadata_sha256": model.metadata_sha256,
        "model_sha256": model.file_sha256,
        "offload_budget_basis": "eligible_tensor_bytes",
        "vmm_capacity_search": {
            "outer_axis": "actual populated context C",
            "inner_axis": "preconfigured weight page budget before model load",
            "algorithm": "fit bracket followed by integer bisection",
            "resolution_bytes": int(manifest["model"]["page_bytes"]),
            "resolution_is_finer_than_0_1pp": True,
            "kv_offload": False,
        },
        "fine_schemes": sweep.get("fine_schemes", list(VMM_SCHEMES)),
        "populated_context": {
            "mechanism": "one llama-bench -pg P,G instance, P+G=C, n_depth=0",
            "minimum_context_tokens": int(manifest["context"].get("minimum_context_tokens", 512)),
            "coarse_step_tokens": int(manifest["context"].get("coarse_step_tokens", 4096)),
            "fine_step_tokens": int(manifest["context"].get("fine_step_tokens", 256)),
            "maximum_context_tokens": int(manifest["context"].get("maximum_context_tokens", 262144)),
            "screen_profile": profile_for(manifest, SCREEN_PROFILE).__dict__,
            "confirmation_profile": profile_for(manifest, CONFIRM_PROFILE).__dict__,
            "warmup": False,
            "host_state_snapshot": False,
        },
        "speed_sources": {
            "prefill": "aggregate formal-rep llama_bench_phase_timing prompt tokens/ns",
            "decode": "aggregate formal-rep llama_bench_phase_timing decode tokens/ns",
            "machine_record": (
                "llama_bench_phase_timing,phase=<prompt|decode>,"
                "rep=<1..R>,tokens=<positive integer>,ns=<positive integer>"
            ),
            "records_required_per_phase": "exactly R, alternating prompt/decode by rep",
            "llama_perf_context_print_is_not_used": True,
            "combined_json_avg_ts_is_not_used": True,
        },
        "baseline_rule": (
            "all-local if it fits; otherwise first successful staging shallow point, "
            "explicitly labeled operational rather than fabricated 0%"
        ),
        "stop_rule": {
            "either_prefill_or_decode_retention_lte": float(sweep.get("stop_retention", 0.10)),
            "requires_confirm_profile": True,
        },
        "remote_files_created": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/gemma-q8-context-probes.csv"))
    parser.add_argument(
        "--frontier-output", type=Path, default=Path("results/gemma-q8-context-frontiers.csv")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-model-wait", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--show-telemetry", action="store_true")
    parser.add_argument(
        "--profile",
        choices=base.PROFILE_MODES,
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
            "disable CUDA Graphs in light mode without enabling timing events "
            "(observer-overhead control)"
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
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--phases",
        help="comma-separated subset of direct_load,staging_coarse,alternate_coarse,fine_0_1pp",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, manifest_hash = load_manifest(args.manifest)
    if args.phases:
        requested = [item.strip() for item in args.phases.split(",") if item.strip()]
        unknown = set(requested) - set(PHASES)
        if unknown:
            raise ValueError(f"unknown phases: {sorted(unknown)}")
        manifest["sweep"]["phases"] = [phase for phase in PHASES if phase in requested]
    if args.profile_force_direct:
        manifest.setdefault("profiling", {})["force_direct"] = True
    if args.profile_backend_off:
        if args.profile != "deep":
            raise ValueError("--profile-backend-off requires --profile deep")
        manifest.setdefault("profiling", {})["backend_enabled"] = False
    if args.profile_no_pcie_dmon:
        manifest.setdefault("profiling", {})[f"{args.profile or 'light'}_pcie_dmon"] = False
    model = wait_for_model(manifest, dry_run=args.dry_run, skip_wait=args.skip_model_wait)
    print(json.dumps(plan_summary(manifest, model), indent=2, sort_keys=True), flush=True)
    with Q8SweepRunner(
        manifest,
        manifest_hash,
        args.output,
        args.frontier_output,
        model,
        dry_run=args.dry_run,
        retry_errors=args.retry_errors,
        show_telemetry=args.show_telemetry,
        max_cases=args.max_cases,
        profile_mode=args.profile,
        profile_output=args.profile_output,
        profile_raw=args.profile_raw,
    ) as runner:
        runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
