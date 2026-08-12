# Gemma-4 31B Q8 populated-context experiment

This is a separate experiment from the running Q4 capacity sweep. It uses
`gemma_q8_context_sweep.py`, its own manifest, and its own result files; it does
not read or modify `results/gemma-capacity-sweep.csv`.

## What is being proved

Every successful capacity point is one combined `llama-bench -pg P,G` test
with `P + G = C`. It really prefills P tokens and then decodes G tokens on the
same KV cache. `--ctx-size C` is only the matching allocation ceiling; neither
an allocation-only success nor two independent `-p`/`-n` tests count as a
capacity result.

The screen profile uses `G=16, R=1`, `batch=ubatch=32`. A possible 10% crossing is rerun with
`G=64, R=3` before stopping. Both profiles explicitly use `-p 0 -n 0 -d 0`
and `--no-warmup`, so there are no unintended default tests, duplicated long
prompts, or depth-state snapshots. The benchmark wraps each formal
`test_prompt` and `test_gen` call with a monotonic clock after synchronization
and writes one strict stderr record per phase and repetition:

```text
llama_bench_phase_timing,phase=prompt,rep=1,tokens=240,ns=123456789
llama_bench_phase_timing,phase=decode,rep=1,tokens=16,ns=234567890
```

The runner requires exactly `R` prompt and `R` decode records in alternating
repetition order, validates their token counts and positive integer nanoseconds,
then computes each rate from aggregate tokens divided by aggregate nanoseconds.
Warmup calls never emit these records. `llama_perf_context_print` and the
combined JSON `avg_ts` are not used as speed sources.

The model kernels execute on the GPU. In both VMM schemes the CPU does not
evaluate transformer layers. It provides source storage and launches DMA in
staging mode; zero-copy kernels read mapped host pages directly.

## Fixed execution order

1. Wait until the Q8 file exists, its `.aria2` sidecar is gone, its GGUF
   metadata and tensor table parse successfully, and size/mtime/tensor counts
   are unchanged for three polls.
2. Start all-local at requested `C=256`, record llama.cpp's actual `n_ctx`, and
   increase by 256 until the first OOM or isolated CUDA/MMQ abort. Every case is
   a fresh process, so an abort becomes a recorded capacity failure rather than
   poisoning the next probe.
3. Sweep double-buffer DMA staging by context. For every C, the page budget is
   configured before model load, then bracketed and integer-bisected to the
   exact minimum successful 2 MiB weight page. KV remains on the GPU
   (`-nkvo 0`). The coarse outer context step is 4096 tokens.
4. Repeat the same coarse capacity search for CUDA zero-copy, then stock
   llama.cpp whole-layer CPU offload (`-ngl`).
5. After all coarse paths, refine both VMM paths in 256-token context steps.
   Every C still gets exact page-level bisection; one 2 MiB page is finer than
   the requested 0.1 percentage-point resolution. The run stops only after two
   confirmed consecutive points where either prefill or decode reaches 10%.
   Native offload remains layer-granular.

The stop metric deliberately includes both effects the user experiences:
increasing remote weight traffic and operating at the larger context enabled
by that traffic.

## Correctness and accounting

- Every successful run must emit the expected checksum sequence and report
  zero non-finite logits.
- The first DMA result for each exact `(profile, context)` becomes a strict
  checksum reference. Later DMA and zero-copy results with the same tuple must
  match it exactly. Different contexts have different random-token inputs and therefore
  are not compared bit-for-bit.
- Native whole-layer offload is finite-logit gated, but is not required to be
  bit-identical to the CUDA paths.
- The download gate parses tensor dimensions and GGML types, then sums only
  matrices matched by the Hybrid override. This `eligible_tensor_bytes`, not
  the whole GGUF file length, is the requested percentage-budget basis. The
  CSV separately records requested pages, actual gross remote bytes, staging
  bytes, and net VRAM saved, so requested and realized percentages are not
  conflated.
- If Q8 cannot run at 0%, the first successful 256-token staging point is
  explicitly labeled the operational baseline. The runner never fabricates a
  0% performance number. If all-local C=256 succeeds, it becomes the immutable
  global origin and the direct frontier continues until its first failure.
- The first successful performance origin is marked in the append-only CSV and
  never changes during resume or the fine pass. Zero-copy cannot enter the
  curve without a same-profile, same-C DMA checksum reference; the runner
  automatically materializes that reference when needed.
- Resume validates file size, tensor count, context limit, metadata SHA-256,
  full-file SHA-256, and eligible bytes. A same-sized replacement model cannot
  silently inherit old rows.
- GPU process/device peak, clocks, temperature, power, and utilization are
  sampled throughout model load, prefix population, and the timed probes.

Each row is appended and `fsync`ed locally. The remote receives a shell wrapper
on stdin and creates no scripts, logs, CSVs, or temporary result files.

## Invocation

First update the SSH control-socket path and temporary `LD_LIBRARY_PATH` in
`q8-context-sweep-manifest.rtx5090.json` if the live runtime names changed.
Then validate the plan without SSH:

```bash
python3 gemma_q8_context_sweep.py \
  --manifest q8-context-sweep-manifest.rtx5090.json \
  --dry-run --max-cases 4
```

Run or resume the full ordered experiment:

```bash
python3 gemma_q8_context_sweep.py \
  --manifest q8-context-sweep-manifest.rtx5090.json \
  --output results/gemma-q8-context-probes.csv \
  --frontier-output results/gemma-q8-context-frontiers.csv \
  --retry-errors
```

`gemma-q8-context-probes.csv` is the append-only evidence log, including OOM
brackets. `gemma-q8-context-frontiers.csv` is regenerated locally after every
case and contains the minimum successful page budget for every populated
context.

Useful recovery controls:

```bash
# Run only the direct-load gate after the download is stable.
python3 gemma_q8_context_sweep.py --manifest q8-context-sweep-manifest.rtx5090.json \
  --phases direct_load --max-cases 1

# Resume the final dense pass after all coarse phases are present in the CSV.
python3 gemma_q8_context_sweep.py --manifest q8-context-sweep-manifest.rtx5090.json \
  --phases fine_0_1pp --retry-errors
```

`--skip-model-wait` reduces the gate to one valid status poll; it never skips
the file-exists, minimum-size, or `.aria2` checks.
