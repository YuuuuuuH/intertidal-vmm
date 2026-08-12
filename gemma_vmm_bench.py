#!/usr/bin/env python3
"""Benchmark the page-aliased VMM Intertidal prototype on Gemma 4."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

from gguf import GGUFReader


TENSOR_RE = re.compile(
    r"blk\.(\d+)\.(attn_(k|v|q|output)|ffn_(down|gate|up))\.weight"
)


def tensor_table(model: Path):
    reader = GGUFReader(str(model))
    total = sum(int(t.n_bytes) for t in reader.tensors)
    by_layer: dict[int, list[tuple[str, int]]] = {}
    for tensor in reader.tensors:
        match = TENSOR_RE.fullmatch(tensor.name)
        if match is None or int(tensor.n_bytes) < 4 * 1024 * 1024:
            continue
        by_layer.setdefault(int(match.group(1)), []).append(
            (tensor.name, int(tensor.n_bytes))
        )
    for tensors in by_layer.values():
        tensors.sort(key=lambda item: (-item[1], item[0]))
    return total, by_layer


def balanced_select(by_layer: dict[int, list[tuple[str, int]]], count: int):
    layers = sorted(by_layer)
    selected: list[tuple[str, int]] = []
    rank = 0
    while len(selected) < count:
        candidates = [layer for layer in layers if rank < len(by_layer[layer])]
        if not candidates:
            raise RuntimeError(f"only {len(selected)} eligible tensors for {count} pages")
        remaining = count - len(selected)
        if remaining >= len(candidates):
            chosen_layers = candidates
        else:
            # Pack a partial rank into one alternating staging parity whenever
            # possible.  For 260 pages this produces per-layer counts 5/4 and
            # slots 10+8 MiB, saving 2 MiB of staging without reducing overlap.
            one_parity = candidates[::2]
            if remaining <= len(one_parity):
                indices = [round(i * (len(one_parity) - 1) / max(1, remaining - 1)) for i in range(remaining)]
                chosen_layers = [one_parity[i] for i in indices]
            else:
                indices = [round(i * (len(candidates) - 1) / max(1, remaining - 1)) for i in range(remaining)]
                chosen_layers = [candidates[i] for i in indices]
        selected.extend(by_layer[layer][rank] for layer in chosen_layers)
        rank += 1
    return selected[:count]


def override_pattern(selected: list[tuple[str, int]]) -> str:
    names = [name.replace(".", "[.]") for name, _ in selected]
    return "^(" + "|".join(names) + ")$=CUDA_Hybrid"


def parse_bench(stdout: str):
    lines = [line for line in stdout.splitlines() if line]
    header = next(csv.reader([next(line for line in lines if line.startswith("build_commit,"))]))
    records = [
        dict(zip(header, next(csv.reader([line]))))
        for line in lines
        if line.startswith('"')
    ]
    prompt = next(row for row in records if int(row["n_prompt"]) > 0)
    decode = next(row for row in records if int(row["n_gen"]) > 0)
    return {
        "prefill_tok_s": float(prompt["avg_ts"]),
        "prefill_std_tok_s": float(prompt["stddev_ts"]),
        "decode_tok_s": float(decode["avg_ts"]),
        "decode_std_tok_s": float(decode["stddev_ts"]),
    }


def run_case(args, label: str, pattern: str | None, pages: int | None):
    env = dict(os.environ)
    # The host account injects an unrelated VRAM interception library globally.
    # It both changes CUDA allocation behavior and crashes the gguf Python
    # extension, so formal Intertidal measurements must not inherit it.
    env.pop("LD_PRELOAD", None)
    if pages is not None:
        env["GGML_CUDA_HYBRID_PAGES"] = str(pages)
        env["GGML_CUDA_HYBRID_FRACTION"] = "0.000001"
    command = [
        str(args.bench), "-m", str(args.model),
        "-p", str(args.prompt), "-n", str(args.generation),
        "-r", str(args.repetitions), "-t", str(args.threads),
        "-b", str(args.prompt), "-ub", str(args.ubatch),
        "-fa", "on", "-mmp", "1", "-o", "csv",
    ]
    if pattern is not None:
        command.extend(["-ot", pattern])
    started = time.monotonic()
    proc = subprocess.run(command, env=env, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"{label} failed ({proc.returncode}):\n{proc.stderr[-8000:]}")
    result = parse_bench(proc.stdout)
    result.update(label=label, elapsed_s=time.monotonic() - started)
    for line in proc.stderr.splitlines():
        if line.startswith("hybrid_vmm,"):
            result["vmm_layout"] = line
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=266)
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--generation", type=int, default=64)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-local-control", action="store_true")
    parser.add_argument(
        "--cases", default="original,vmm",
        help="comma-separated subset of original,local,vmm",
    )
    args = parser.parse_args()

    total, by_layer = tensor_table(args.model)
    selected = balanced_select(by_layer, args.pages)
    pattern = override_pattern(selected)
    per_layer: dict[int, int] = {}
    for name, _ in selected:
        layer = int(name.split(".")[1])
        per_layer[layer] = per_layer.get(layer, 0) + 1
    metadata = {
        "model_bytes": total,
        "pages": args.pages,
        "gross_remote_bytes": args.pages * 2 * 1024 * 1024,
        "tensor_count": len(selected),
        "layers": len(per_layer),
        "min_pages_per_layer": min(per_layer.values()),
        "max_pages_per_layer": max(per_layer.values()),
        "selected": [name for name, _ in selected],
    }
    print(json.dumps(metadata, sort_keys=True), flush=True)

    cases = {item.strip() for item in args.cases.split(",") if item.strip()}
    results = []
    if "original" in cases:
        results.append(run_case(args, "original_vram", None, None))
    if "local" in cases and not args.skip_local_control:
        results.append(run_case(args, "vmm_all_local", pattern, 0))
    if "vmm" in cases:
        results.append(run_case(args, "vmm_h2d", pattern, 1))
    baseline = next((item for item in results if item["label"] == "original_vram"), None)
    if baseline is not None:
        for result in results:
            if result is baseline:
                continue
            result["prefill_retention"] = result["prefill_tok_s"] / baseline["prefill_tok_s"]
            result["decode_retention"] = result["decode_tok_s"] / baseline["decode_tok_s"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "results": results}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
