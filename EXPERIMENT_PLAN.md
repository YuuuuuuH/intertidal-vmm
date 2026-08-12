# Gemma-4 fixed-model capacity sweep

## Objective

Measure how far a fixed Gemma-4-31B Q4_0 model can extend the usable GPU
working set by keeping part of its weights in system RAM.  Compare three
implementations on the RTX 5090:

1. `intertidal_dma`: CUDA VMM aliases selected weight pages to two VRAM staging
   slots and overlaps H2D DMA with the original GPU kernels.
2. `cuda_zero_copy`: the same remote-weight byte budget is read directly by GPU
   kernels over PCIe, without CPU compute.
3. `native_cpu_layers`: stock llama.cpp whole-layer CPU offload (`-ngl`), which
   is the conventional fallback and does use the CPU for compute.

The GPU uses NVIDIA/CUDA's default allocation policy.  There is no artificial
VRAM limit, fit margin, clock lock, or reserved safety pool.

## Definitions

- Binary units are used throughout: MiB = 2^20 bytes and GiB = 2^30 bytes.
- The fixed GGUF contains 17,322,412,272 bytes (16.132754 GiB) of tensors.
- One requested 0.1 percentage point of weight offload is about 16.520 MiB.
- The VMM allocation granularity is 2 MiB, so every row records both requested
  and realized gross percentages.
- `net_saved = gross_remote - staging - VMM_rounding_overhead`.
- `logical_working_set = measured_process_GPU_peak + measured_net_saved`.
- Gross remote bytes are never reported as usable capacity without subtracting
  staging and allocator overhead.
- Native `-ngl` has no VMM-style gross byte count.  Its net saving is the
  same-context all-local process peak minus the measured offloaded process
  peak.  If that all-local allocation is above physical VRAM and therefore
  cannot be measured, substitute the calibrated same-context all-local peak
  and mark the row `inferred_calibrated_all_local_peak_delta`.

An earlier claim treated a 513.083 MiB net saving as a percentage multiplier
on the entire GPU working set.  That hypothetical proportional model is not a
measured capacity result and is not valid accounting for fixed Gemma plus a
growing KV cache.  Here the capacity gain is only the measured net number of
bytes released from VRAM.

"Logical" never means additional physical framebuffer.  This RTX 5090 has
32,607 MiB of physical VRAM (about 31.84 GiB), and `nvidia-smi` cannot report
33--34 GiB allocated on it.  A 33 GiB logical proof means that a workload whose
same-context all-local equivalent is 33 GiB succeeds because some fixed model
weights reside in system RAM; the identical-context all-local control must
fail.

## Preliminary server context map

The model has 60 blocks: 50 use a 1024-token sliding window and 10 use global
attention.  With F16 K/V, one server slot, Flash Attention, and full GPU
offload, process VRAM followed this measured relation exactly at 4K, 64K, and
128K contexts:

```text
peak_MiB = 18460 + context_tokens * 81 / 1024
```

Calibration points:

| context | process peak |
| ---: | ---: |
| 4,096 | 18,784 MiB |
| 65,536 | 23,644 MiB |
| 131,072 | 28,828 MiB |
| 171,776 | 32,048 MiB (server ready) |
| 172,032 | allocation failure |

These measurements use `llama-server`.  Its compute buffers are not identical
to `llama-bench`, so this map is used only for the end-to-end server proof.
Before the performance sweep, repeat at least the 4K/64K/128K calibration with
the exact patched `llama-bench` binary and derive a runner-specific intercept.
The 31 GiB point must correspond to the benchmark process actually using about
31 GiB; it must not inherit the server context value by name.

For the server runner, the 31 GiB non-full comparison maps exactly to context
167,936.  Context is rounded to 256-token units for the 31.0--34.0 GiB capacity
grid, and the actual logical MiB is retained rather than relabeling a rounded
point as exact.

Approximate server milestone contexts are:

| requested logical working set | context |
| ---: | ---: |
| 31 GiB | 167,936 |
| 32 GiB | 180,992 |
| 33 GiB | 193,792 |
| 34 GiB | 206,848 |

## Experiment matrix

### Execution priority: characterize the 90--95% sweet spot first

The formal curve still runs to the capacity limit or the 10% retention stop
rule below.  The 90--95% band is an investigation priority, not an early stop
condition and not a replacement baseline.

Historical Q4 DMA data locates the decode-defined band near 3.3--3.8% gross
weight offload: 3.3% retained about 95.14%, 3.4% about 94.93%, 3.8% about
91.02%, and 3.9% about 88.97%.  Prefill remained near 99% in this interval.
These old points are locators only because they did not have bracketing
same-session all-local sentinels.  The old layout's physical staging allocation
also stepped from 20 MiB at 3.6% to 24 MiB at 3.7%, while decode retention fell
from about 94.26% to 92.82%.  Profile this boundary to distinguish a layout or
per-layer-copy knee from unavoidable aggregate PCIe saturation.

On a fresh binary, run the following order:

1. Recalibrate same-boot pinned H2D bandwidth and pass the staging correctness
   gates before collecting performance data.
2. With profiling disabled and CUDA Graphs enabled, scan Q4 DMA from 2.8% to
   4.2% in 0.1-point increments.  Use fresh output files and all-local controls
   before and after the scan.
3. Confirm the 95% and 90% boundaries with ten repetitions.  At minimum retain
   3.3%, 3.4%, 3.6%, 3.7%, 3.8%, and 3.9% as diagnosis points.
4. Profile prefill and decode in separate processes at those selected points.
   Measure light Graph-on telemetry, a direct-stream/profile-off observer
   control, and deep CUDA-event instrumentation.  Deep-profile tok/s is never
   used as production retention.
5. Resume or rerun the complete 0.1-point DMA curve through two consecutive
   points at or below 10%, then run zero-copy, native whole-layer offload, the
   31--34 GiB frontier, and populated-context milestones.

Sweet-spot, profiling, and canonical full-curve runs use distinct CSV/JSONL
files.  A selected profiling row must not seed a production sweep because its
Graph mode, observer overhead, repetition count, and phase identity differ.
Q4 fixed `pp512`/`tg64` retention is the pure weight-offload curve.  Q8
populated-context retention additionally includes longer-attention arithmetic
and is reported separately.

### A. Pure offload-cost curve

- Keep the benchmark work fixed at `pp512` and `tg64`.
- Reserve context independently from processed tokens when a capacity-sized KV
  allocation is needed.
- Sweep requested gross weight offload at 0.1 percentage-point resolution.
- First run a 1.0-point coarse pilot to locate knees and failures, then fill all
  intervening 0.1-point samples.
- Intertidal and zero-copy use a deterministic nested page layout: the pages at
  budget `N` are a strict subset of budget `N+1`.
- Native CPU offload is measured only at real whole-layer points; it is never
  interpolated or described as 0.1% data.

### B. Capacity frontier

- Sweep logical working set from 31.0 to 34.0 GiB in 0.1 GiB increments.
- At each target, find the minimum 0.1-point offload budget that reaches server
  ready under the default NVIDIA allocation policy.
- Record all-local OOM/failure at the identical context when applicable.
- Measure fixed `pp512`/`tg64` work at that reserved context to isolate offload
  cost from the arithmetic cost of a longer prompt.

### C. End-to-end capacity proof

- At 31, 32, 33, and 34 GiB milestones, actually populate a long prompt close
  to the reserved context and perform decode.
- At over-physical points, require the same-context all-local run to fail and
  the offloaded run to complete with finite logits.
- Recheck representative output/logit agreement and CUDA Graph launches.

## Benchmark controls

- Model, llama.cpp commit, compiler, CUDA version, Flash Attention, KV types,
  mmap policy, batch, ubatch, CPU binding, and GPU driver remain fixed.
- Remove the unrelated global `LD_PRELOAD=libvramctl_preload.so` from every
  formal run.
- Use two warmups and five recorded repetitions per point.  Repeat points whose
  coefficient of variation exceeds 2%; use ten repetitions at knees and the
  95%/10% crossings.
- Insert an all-local sentinel at least every 1.0 percentage point and normalize
  to the nearest same-session sentinel.  Retain the historical
  `2262.12 pp512 tok/s` and `75.48 tg tok/s` only as an audit reference.
- Treat 95% and 90% as diagnostic boundaries.  A production point in the
  90--95% band is acceptable for capacity-oriented operation, but it does not
  change the full-curve stopping rule.
- Record process peak VRAM, graphics clock, temperature, power, GPU utilization,
  host `MemAvailable`, major faults, and memory PSI.
- Stop a curve after reaching 34 GiB, or after either prefill or decode remains
  at or below 10% of its same-session initial baseline for two consecutive
  samples.  Correctness failure stops the run immediately and is not a slow
  datapoint.

## Native whole-layer baseline

Sweep `-ngl 61,60,59,...`.  Gemma-4 has 60 transformer blocks plus the output
layer, so `-ngl 60` moves only `blk.0` to CPU.  The CPU-mapped model buffer has
a fixed component even at `-ngl 61`; native released bytes are therefore the
difference from the same-context `-ngl 61` peak, cross-checked against loader
buffer accounting.  The frontier does not accept the first successful
`-ngl`: it also requires the measured or explicitly inferred net saving to
cover `calibrated_all_local_peak - default_fit_mib`, then records the next
whole-layer point as confirmation.

## Remote-host discipline

The 5090 host receives only a temporary source/build tree for the duration of
the run.  Result rows stream back to this local private repository as they are
completed.  Runs are resumable.  At the end, remove the remote build, logs,
core files, and temporary results, and verify that no benchmark process or GPU
allocation remains.

## Deliverables

- Raw append-only CSV/JSONL data and a machine-readable run manifest.
- Two-panel prefill/decode plots with tok/s, retention, confidence intervals,
  95% and 10% reference lines, OOM markers, and measured logical capacity.
- A separate capacity-frontier plot from 31.0 through 34.0 GiB.
- Exact commands, environment, source commit, patch checksum, and cleanup
  audit committed to the private repository.
