#!/usr/bin/env python3
"""Benchmark Gemma 4 with selected Q4 attention weights in mapped host RAM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from gguf import GGUFReader


def target_values(max_pct: float, step_pct: float) -> list[float]:
    count = round(max_pct / step_pct)
    return [round(i * step_pct, 6) for i in range(count + 1)]


def tensor_sets(model: Path, targets: list[float]):
    reader = GGUFReader(str(model))
    total_bytes = sum(int(t.n_bytes) for t in reader.tensors)
    candidates = [
        t
        for t in reader.tensors
        if re.fullmatch(r"blk\.\d+\.attn_[kv]\.weight", t.name)
    ]
    candidates.sort(
        key=lambda t: hashlib.sha256(t.name.encode("utf-8")).digest()
    )

    chosen = []
    remaining = list(candidates)
    chosen_bytes = 0
    result = {}

    for pct in targets:
        target = total_bytes * pct / 100.0
        while remaining:
            current_error = abs(chosen_bytes - target)
            best = min(
                remaining,
                key=lambda t: (
                    abs(chosen_bytes + int(t.n_bytes) - target),
                    hashlib.sha256(t.name.encode("utf-8")).digest(),
                ),
            )
            next_error = abs(chosen_bytes + int(best.n_bytes) - target)
            if next_error >= current_error:
                break
            chosen.append(best)
            remaining.remove(best)
            chosen_bytes += int(best.n_bytes)
        result[pct] = (list(chosen), chosen_bytes, total_bytes)

    return result


def override_pattern(tensors) -> str | None:
    if not tensors:
        return None
    names = [t.name.replace(".", "[.]") for t in tensors]
    return "^(" + "|".join(names) + ")$=CUDA_Host"


def parse_bench(stdout: str):
    lines = [line for line in stdout.splitlines() if line]
    header_line = next(line for line in lines if line.startswith("build_commit,"))
    header = next(csv.reader([header_line]))
    records = []
    for line in lines:
        if not line.startswith('"'):
            continue
        values = next(csv.reader([line]))
        records.append(dict(zip(header, values)))
    prefill = next(r for r in records if int(r["n_prompt"]) > 0)
    decode = next(r for r in records if int(r["n_gen"]) > 0)
    return prefill, decode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-pct", type=float, default=5.0)
    parser.add_argument("--step-pct", type=float, default=0.1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--generation", type=int, default=64)
    parser.add_argument("--ubatch", type=int, default=16)
    args = parser.parse_args()

    targets = target_values(args.max_pct, args.step_pct)
    sets = tensor_sets(args.model, targets)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.environ.get("INTERTIDAL_LD_LIBRARY_PATH", "")
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"

    fieldnames = [
        "target_pct",
        "actual_pct",
        "remote_bytes",
        "remote_MiB",
        "tensor_count",
        "prefill_tok_s",
        "prefill_std_tok_s",
        "decode_tok_s",
        "decode_std_tok_s",
        "elapsed_s",
        "status",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists() and args.output.stat().st_size > 0
    completed = set()
    if exists:
        with args.output.open(newline="") as old:
            completed = {float(row["target_pct"]) for row in csv.DictReader(old) if row["status"] == "ok"}

    with args.output.open("a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
            out.flush()

        for pct in targets:
            if pct in completed:
                print(f"skip target={pct:.1f}% (already complete)", flush=True)
                continue
            tensors, remote_bytes, total_bytes = sets[pct]
            actual_pct = 100.0 * remote_bytes / total_bytes
            command = [
                str(args.bench),
                "-m", str(args.model),
                "-p", str(args.prompt),
                "-n", str(args.generation),
                "-r", str(args.repetitions),
                "-t", "8",
                "-b", str(args.prompt),
                "-ub", str(args.ubatch),
                "-fa", "on",
                "-mmp", "0",
                "-o", "csv",
            ]
            pattern = override_pattern(tensors)
            if pattern:
                command.extend(["-ot", pattern])

            print(
                f"start target={pct:.1f}% actual={actual_pct:.4f}% "
                f"remote={remote_bytes / 2**20:.2f} MiB tensors={len(tensors)}",
                flush=True,
            )
            started = time.monotonic()
            proc = subprocess.run(command, env=env, text=True, capture_output=True)
            elapsed = time.monotonic() - started
            row = {
                "target_pct": f"{pct:.1f}",
                "actual_pct": f"{actual_pct:.6f}",
                "remote_bytes": remote_bytes,
                "remote_MiB": f"{remote_bytes / 2**20:.3f}",
                "tensor_count": len(tensors),
                "prefill_tok_s": "",
                "prefill_std_tok_s": "",
                "decode_tok_s": "",
                "decode_std_tok_s": "",
                "elapsed_s": f"{elapsed:.3f}",
                "status": "ok",
            }
            try:
                if proc.returncode != 0:
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
                print(proc.stderr[-4000:], file=sys.stderr, flush=True)
            writer.writerow(row)
            out.flush()
            print(
                f"done target={pct:.1f}% pp={row['prefill_tok_s'] or 'FAIL'} "
                f"tg={row['decode_tok_s'] or 'FAIL'} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if row["status"] != "ok":
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
