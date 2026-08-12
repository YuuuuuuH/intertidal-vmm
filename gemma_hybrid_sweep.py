#!/usr/bin/env python3
"""Sweep model-wide RAM percentages for the CUDA double-buffer staging prototype."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from gguf import GGUFReader


TENSOR_RE = re.compile(
    r"blk\.\d+\.(attn_(k|v|q|output)|ffn_(down|gate|up))\.weight"
)
ALL_OVERRIDE = (
    r"^blk[.][0-9]+[.](attn_(k|v|q|output)|ffn_(down|gate|up))"
    r"[.]weight$=CUDA_Hybrid"
)


def f32(value: float) -> float:
    return ctypes.c_float(value).value


def tensor_table(model: Path):
    reader = GGUFReader(str(model))
    total_bytes = sum(int(t.n_bytes) for t in reader.tensors)
    covered = []
    for tensor in reader.tensors:
        if not TENSOR_RE.fullmatch(tensor.name):
            continue
        if len(tensor.shape) != 2:
            raise RuntimeError(f"expected 2-D tensor: {tensor.name} {tensor.shape}")
        nrows = int(tensor.shape[1])
        nbytes = int(tensor.n_bytes)
        if nbytes % nrows:
            raise RuntimeError(f"non-integral row size: {tensor.name}")
        covered.append((tensor.name, nrows, nbytes // nrows))
    return total_bytes, covered


def remote_bytes_for_fraction(covered, fraction: float) -> int:
    fraction = f32(fraction)
    result = 0
    for _, nrows, row_size in covered:
        remote_rows = math.ceil(nrows * fraction)
        remote_rows = max(1, min(nrows - 1, remote_rows))
        result += remote_rows * row_size
    return result


def remote_bytes_by_tensor(covered, fraction: float):
    fraction = f32(fraction)
    result = []
    for item in covered:
        name, nrows, row_size = item
        remote_rows = max(1, min(nrows - 1, math.ceil(nrows * fraction)))
        result.append((item, remote_rows * row_size))
    return result


def select_tensors(covered, fraction: float, target_bytes: float):
    remaining = remote_bytes_by_tensor(covered, fraction)
    remaining.sort(key=lambda pair: hashlib.sha256(pair[0][0].encode()).digest())
    chosen = []
    chosen_bytes = 0
    while remaining:
        current_error = abs(chosen_bytes - target_bytes)
        best = min(
            remaining,
            key=lambda pair: (
                abs(chosen_bytes + pair[1] - target_bytes),
                hashlib.sha256(pair[0][0].encode()).digest(),
            ),
        )
        if abs(chosen_bytes + best[1] - target_bytes) >= current_error:
            break
        chosen.append(best[0])
        chosen_bytes += best[1]
        remaining.remove(best)
    return chosen, chosen_bytes


def select_tail_tensors(covered, fraction: float, target_bytes: float):
    by_layer = {}
    for item, nbytes in remote_bytes_by_tensor(covered, fraction):
        layer = int(item[0].split(".")[1])
        by_layer.setdefault(layer, []).append((item, nbytes))

    chosen = []
    chosen_bytes = 0
    for layer in sorted(by_layer, reverse=True):
        group = sorted(by_layer[layer], key=lambda pair: pair[0][0])
        group_bytes = sum(pair[1] for pair in group)
        if abs(chosen_bytes + group_bytes - target_bytes) <= abs(chosen_bytes - target_bytes):
            chosen.extend(pair[0] for pair in group)
            chosen_bytes += group_bytes
            continue

        # Preserve a contiguous tail of complete layers; use only the boundary
        # layer to trim the final capacity error.
        remaining = list(group)
        while remaining:
            current_error = abs(chosen_bytes - target_bytes)
            best = min(
                remaining,
                key=lambda pair: (abs(chosen_bytes + pair[1] - target_bytes), pair[0][0]),
            )
            if abs(chosen_bytes + best[1] - target_bytes) >= current_error:
                break
            chosen.append(best[0])
            chosen_bytes += best[1]
            remaining.remove(best)
        break
    return chosen, chosen_bytes


def override_pattern(selected, all_tensors: bool) -> str:
    if all_tensors:
        return ALL_OVERRIDE
    names = [item[0].replace(".", "[.]") for item in selected]
    return "^(" + "|".join(names) + ")$=CUDA_Hybrid"


def solve_fraction(covered, target_bytes: float) -> tuple[float, int]:
    lo, hi = 0.0, 0.95
    for _ in range(60):
        mid = f32((lo + hi) / 2)
        if remote_bytes_for_fraction(covered, mid) < target_bytes:
            lo = mid
        else:
            hi = mid
    candidates = {f32(lo), f32(hi)}
    # Move a few float32 ULPs on both sides of the discontinuity.
    for value in list(candidates):
        for direction in (-math.inf, math.inf):
            moved = value
            for _ in range(4):
                moved = f32(math.nextafter(moved, direction))
                candidates.add(moved)
    fraction = min(
        (v for v in candidates if v > 0),
        key=lambda v: (abs(remote_bytes_for_fraction(covered, v) - target_bytes), v),
    )
    return fraction, remote_bytes_for_fraction(covered, fraction)


def parse_bench(stdout: str):
    lines = [line for line in stdout.splitlines() if line]
    header_line = next(line for line in lines if line.startswith("build_commit,"))
    header = next(csv.reader([header_line]))
    records = []
    for line in lines:
        if line.startswith('"'):
            records.append(dict(zip(header, next(csv.reader([line])))))
    return (
        next(row for row in records if int(row["n_prompt"]) > 0),
        next(row for row in records if int(row["n_gen"]) > 0),
    )


def targets_from_args(args) -> list[float]:
    if args.points:
        return sorted({round(float(value), 6) for value in args.points.split(",")})
    count = round((args.max_pct - args.min_pct) / args.step_pct)
    return [round(args.min_pct + i * args.step_pct, 6) for i in range(count + 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-pct", type=float, default=0.0)
    parser.add_argument("--max-pct", type=float, default=5.0)
    parser.add_argument("--step-pct", type=float, default=0.1)
    parser.add_argument("--points", help="comma-separated targets; overrides min/max/step")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--generation", type=int, default=64)
    parser.add_argument("--ubatch", type=int, default=16)
    parser.add_argument("--selection", choices=("spread", "tail", "last-layer"), default="spread")
    args = parser.parse_args()

    targets = targets_from_args(args)
    total_bytes, covered = tensor_table(args.model)
    covered_bytes = sum(nrows * row_size for _, nrows, row_size in covered)
    print(
        f"model={total_bytes / 2**30:.3f} GiB covered={covered_bytes / 2**30:.3f} GiB "
        f"({100 * covered_bytes / total_bytes:.3f}%) tensors={len(covered)}",
        flush=True,
    )

    env_base = dict(os.environ)
    env_base["LD_LIBRARY_PATH"] = os.environ.get("INTERTIDAL_LD_LIBRARY_PATH", "")
    env_base["GGML_CUDA_DISABLE_GRAPHS"] = "1"

    fields = [
        "target_pct", "actual_pct", "selection", "hybrid_fraction", "remote_bytes", "remote_MiB",
        "covered_bytes", "tensor_count", "prefill_tok_s", "prefill_std_tok_s",
        "decode_tok_s", "decode_std_tok_s", "elapsed_s", "status",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists() and args.output.stat().st_size > 0
    completed = set()
    if exists:
        with args.output.open(newline="") as old:
            completed = {
                float(row["target_pct"])
                for row in csv.DictReader(old)
                if row["status"] == "ok"
            }

    with args.output.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if not exists:
            writer.writeheader()
            output.flush()

        for target_pct in targets:
            if target_pct in completed:
                print(f"skip target={target_pct:.1f}%", flush=True)
                continue

            selected = []
            all_tensors = False
            if target_pct == 0:
                fraction, remote_bytes = 0.0, 0
            else:
                target_bytes = total_bytes * target_pct / 100.0
                if args.selection == "last-layer":
                    last_layer = max(int(item[0].split(".")[1]) for item in covered)
                    selected = [item for item in covered if int(item[0].split(".")[1]) == last_layer]
                    if target_bytes >= sum(nrows * row_size for _, nrows, row_size in selected):
                        raise RuntimeError("target exceeds the last layer's covered weight capacity")
                    fraction, remote_bytes = solve_fraction(selected, target_bytes)
                else:
                    stripe_fraction = f32(0.03)
                    stripe_capacity = remote_bytes_for_fraction(covered, stripe_fraction)
                    if target_bytes <= stripe_capacity:
                        fraction = stripe_fraction
                        if args.selection == "tail":
                            selected, remote_bytes = select_tail_tensors(covered, fraction, target_bytes)
                        else:
                            selected, remote_bytes = select_tensors(covered, fraction, target_bytes)
                    else:
                        fraction, remote_bytes = solve_fraction(covered, target_bytes)
                        selected = list(covered)
                        all_tensors = True
            actual_pct = 100.0 * remote_bytes / total_bytes
            env = dict(env_base)
            if fraction:
                env["GGML_CUDA_HYBRID_FRACTION"] = f"{fraction:.9g}"

            command = [
                str(args.bench), "-m", str(args.model),
                "-p", str(args.prompt), "-n", str(args.generation),
                "-r", str(args.repetitions), "-t", "8",
                "-b", str(args.prompt), "-ub", str(args.ubatch),
                "-fa", "on", "-mmp", "1", "-o", "csv",
            ]
            if fraction:
                command.extend(["-ot", override_pattern(selected, all_tensors)])

            print(
                f"start target={target_pct:.1f}% actual={actual_pct:.6f}% "
                f"fraction={fraction:.9g} remote={remote_bytes / 2**20:.3f} MiB",
                flush=True,
            )
            started = time.monotonic()
            proc = subprocess.run(command, env=env, text=True, capture_output=True)
            elapsed = time.monotonic() - started
            row = {
                "target_pct": f"{target_pct:.1f}",
                "actual_pct": f"{actual_pct:.9f}",
                "selection": args.selection,
                "hybrid_fraction": f"{fraction:.9g}",
                "remote_bytes": remote_bytes,
                "remote_MiB": f"{remote_bytes / 2**20:.6f}",
                "covered_bytes": covered_bytes,
                "tensor_count": len(selected),
                "prefill_tok_s": "", "prefill_std_tok_s": "",
                "decode_tok_s": "", "decode_std_tok_s": "",
                "elapsed_s": f"{elapsed:.3f}", "status": "ok",
            }
            try:
                if proc.returncode:
                    raise RuntimeError(f"exit {proc.returncode}")
                prefill, decode = parse_bench(proc.stdout)
                row.update(
                    prefill_tok_s=f'{float(prefill["avg_ts"]):.6f}',
                    prefill_std_tok_s=f'{float(prefill["stddev_ts"]):.6f}',
                    decode_tok_s=f'{float(decode["avg_ts"]):.6f}',
                    decode_std_tok_s=f'{float(decode["stddev_ts"]):.6f}',
                )
            except Exception as exc:
                row["status"] = str(exc)
                print(proc.stderr[-6000:], file=sys.stderr, flush=True)
            writer.writerow(row)
            output.flush()
            print(
                f"done target={target_pct:.1f}% pp={row['prefill_tok_s'] or 'FAIL'} "
                f"tg={row['decode_tok_s'] or 'FAIL'} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if row["status"] != "ok":
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
