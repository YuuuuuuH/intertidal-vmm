# PCIe weight striping experiment

Measured on the RTX 5090 test host on 2026-08-11 and 2026-08-12.

## Validated VMM double-buffer staging point

The VMM path has passed correctness and low-offload performance validation on
the RTX 5090 with Gemma-4-31B Q4_0.  This historical measurement is a
fixed-workload performance point, not yet the final 31--34 GiB capacity-frontier
result.  The path is not the early row-split prototype described later in this
document.  It uses CUDA VMM to give every weight tensor a stable, contiguous
virtual address:

- ordinary pages map to private VRAM;
- selected complete 2 MiB pages alias one of two physical staging slots;
- remote pages are packed by transformer layer in pinned RAM;
- layer `L+1` is copied on a non-blocking stream while layer `L` computes;
- the last layer ring-prefetches layer zero for the next CUDA Graph launch;
- original MMQ/MMVQ, stream-K, fusions, and CUDA Graphs remain enabled.

There is no special remote matrix kernel and no output merge.  The original
kernel dereferences the original tensor pointer after the staging physical
pages have been filled.

Validated point, 10 repetitions, `pp512`, `tg64`, `ubatch=512`, Flash
Attention on:

| Case | Prefill tok/s | Decode tok/s | Retention | Net VRAM released |
| --- | ---: | ---: | ---: | ---: |
| Original all-VRAM (post control) | 2262.12 | 75.48 | 100% | 0 |
| VMM + H2D, 266 pages | 2190.11 | 73.00 | 96.82% / 96.72% | 513.083 MiB |

The staged layout was 532 MiB of remote weight pages across all 60 layers and
only 18 MiB of physical staging VRAM.  Including VMM end-page rounding, the
measured net saving was 513.083 MiB out of a 16.133 GiB model, or 3.1058% of
the model-weight footprint.  Prefill and decode retained 96.82% and 96.72% at
this point.  The saving is additive for fixed Gemma plus KV cache: it releases
513.083 MiB for additional KV or other allocations.  It must not be treated as
a percentage multiplier on the entire GPU working set, and this point by
itself does not establish a 32-to-33 GiB fixed-model capacity result.

The distinction is important: the card's physical framebuffer remains the
32,607 MiB total reported by `nvidia-smi` (about 31.84 GiB), and
`nvidia-smi` will never show 33 GiB or more allocated on this card.  A reported
logical all-local-equivalent working set is instead calculated per run as
`measured hybrid process GPU peak + measured net VRAM saving`.  Remote weight
bytes reside in system RAM.  A valid over-physical capacity proof therefore
requires the hybrid run to succeed while an identical-context all-local run
fails; it is not an increase in physical framebuffer size.

The historical all-local VMM control reached 76.27 decode tok/s versus 75.48
for the original allocator, showing that stable virtual addressing and the
original compute kernels add no measurable decode tax.  Prefill on this host
has large run-to-run thermal/clock variance; this H2D point uses the later,
higher original baseline as the conservative denominator.  The resumable
fixed-model sweep below is the authoritative source for the capacity
conclusion once it is complete.

Correctness was checked after the 532 MiB run.  The complete prompt/decode
logit dumps were byte-identical to all-VRAM and had the same SHA-256:

```text
af1fffc617a8ead1f051d79c0a64a485784946e1d15734968351d25f4d91561e
```

A separate adversarial audit then checked the two assumptions that are easiest
to fake accidentally:

- `nvidia-smi` sampled the process every 100 ms during identical 512-token
  runs.  Peak process VRAM was 18,080 MiB all-local and 17,566 MiB with VMM: an
  externally observed 514 MiB reduction versus the VMM page accounting of
  513.083 MiB.
- An opt-in internal audit recorded 2 CUDA Graph captures and 65
  `cudaGraphLaunch` calls in the 64-token test.  The Hybrid path was therefore
  replaying CUDA Graphs, not silently using ordinary launches.
- The full 64-token logit dumps were both 4.0 MiB, byte-identical, and had
  SHA-256 `799e4d9029f1db9069d1b13b452e3d7c4085b71048994e7dc35cc4572a12305a`.

Artifacts:

- `intertidal-vmm.patch`: patch against llama.cpp commit `b820cc8`.
- `ggml-cuda-hostmapped.cu`: complete experimental CUDA source.
- `llama-bench-host.cpp`: complete benchmark source, including an independent
  `--ctx-size` KV-reservation control.
- `gemma_vmm_bench.py`: balanced page selector and three-case runner.
- `vmm-ring2-final-266-r10.json`: final 266-page measurements.
- `vmm-final-original-post-r10.json`: post-run all-VRAM control.
- `vmm-final-local-r10.json`: all-local VMM control.
- `vmm-hybrid-266.err`: layout and final correctness checksums.
- `intertidal-audit.txt`: independent VRAM, CUDA Graph, and 64-token audit.
- `PROFILING.md`: PCIe/DMA latency, workload, observer-control, calibration,
  and Nsight diagnosis runbook.
- `pcie_h2d_calibrate.cu`: matched-size pinned-H2D latency and bandwidth
  calibration used to convert measured DMA GB/s into PCIe utilization.
- `analyze_profile.py`: local JSONL-to-CSV/Markdown/SVG diagnosis report with
  per-layer mixed-size PCIe-utilization accounting.

The login environment on the test host globally set an unrelated
`LD_PRELOAD=...libvramctl_preload.so`.  Formal results explicitly removed it;
leaving it enabled changes CUDA allocation behavior and even crashes the local
Python GGUF import.

## Test platform

- NVIDIA GeForce RTX 5090, 32,607 MiB, driver 580.159.03
- PCIe 5.0 x16 negotiated at 32 GT/s x16
- 32 GiB Resizable BAR
- Intel Core Ultra 5 230F, 31 GiB system RAM
- CUDA H2D pinned-memory copy: 48.73 GB/s
- CUDA D2H pinned-memory copy: 22.84 GB/s
- GPU-local scan kernel: about 1,700 GB/s
- Gemma-4-31B Q4_0 (16.13 GiB) decode: 71.84 tok/s, or about 1.24 TB/s of
  effective model-weight traffic

The bandwidth-only equations are:

```text
optimal remote fraction of total weights = B_pcie / (B_gpu + B_pcie)
ideal extra remote weight bytes          = W_local * B_pcie / B_gpu
remote-bound relative throughput         = B_pcie / (remote_fraction * B_gpu)
```

The middle expression is only a bandwidth model for scalable weight bytes.  It
is not the accounting rule for this fixed Gemma model with a growing KV cache.
Fixed-model capacity gain is the measured net VRAM saving in bytes.

On this 5090, the DMA-staged optimum is only about 2.8% for the synthetic scan
and about 3.8% for the real Q4 decode. It is therefore a useful worst-case
platform, not the desired 10-20% sweet-spot platform.

## Microbenchmark result

The benchmark scans 8,388,608,000 bytes per iteration. The remote allocation is
mapped pinned host RAM. Blocks are interleaved at 1% granularity. Two modes are
implemented:

- `zero_copy`: the CUDA kernel directly dereferences mapped host memory.
- `dma_staged`: two 128 MiB scratch buffers pipeline H2D DMA and GPU scanning.

| Remote | Zero-copy relative | DMA-staged relative |
| ---: | ---: | ---: |
| 0% | 100.0% | 100.0% |
| 1% | 101.0% | 98.8% |
| 2% | 82.1% | 97.7% |
| 3% | 44.9% | 95.6% |
| 4% | 32.7% | 72.0% |
| 5% | 25.6% | 57.7% |
| 10% | 11.5% | 28.8% |
| 15% | 7.0% | 19.3% |
| 20% | 5.0% | 14.4% |
| 25% | 3.9% | 11.6% |
| 30% | 3.2% | 9.6% |
| 40% | 2.3% | 7.2% |

Direct reads only sustained roughly 16-22 GB/s even though H2D DMA reached
48.73 GB/s. Write-combined host allocation and additional load ILP did not
materially improve zero-copy. A practical implementation should therefore begin
with DMA staging, not direct mapped-memory loads.

The 1-3% sweep was repeated. DMA staging was stable: 2% retained 97.7% in both
runs, while 3% retained 95.6% and 94.4%. Direct zero-copy varied more at 2%
(about 80-86%) and collapsed beyond that point. The staged knee is sharply
between 3% and 4%, matching the 2.8% bandwidth-ratio prediction closely.

At a temporary 600 MHz graphics-clock lock, local scanning fell to about
703 GB/s and H2D fell to about 37 GB/s. DMA-staged 20% and 30% retained 24.5%
and 17.0%, respectively. Lowering the core clock is not a clean simulation of a
400 GB/s decode kernel because it also lowers PCIe copy throughput.

## Real-model control

The existing 16.13 GiB Gemma-4-31B Q4_0 is a good capacity-cliff proxy for a
16 GiB card:

- Full GPU: 71.84 tok/s.
- llama.cpp constrained to about a 16 GiB device budget: 14.84 tok/s (20.7%).
- The constrained load placed 15,200.53 MiB of model buffers on CUDA and
  2,075.48 MiB in CPU-mapped memory, with 56/61 layers on GPU.

This is the serial/layer-offload control. It is not the proposed row-striped
implementation.

## Superseded row-split implementation plan

The following was the original plan before the VMM design was implemented.  It
is retained only as experiment history; the final implementation above avoids
its kernel and scheduling taxes.

1. Split only large Q4 linear tensors by output rows. Keep norms, embeddings,
   small tensors, KV cache, and recurrent state in VRAM.
2. Keep the remote rows in pinned file-backed RAM. Do not duplicate the whole
   GGUF into a second host allocation.
3. Use two 64-128 MiB VRAM staging buffers. Pipeline H2D copies with the existing
   local `mmvq` kernel, and write remote rows directly into their output slice.
   A row split avoids a partial-sum reduction.
4. Sweep 0, 1, 2, 3, 4, 5, 8, 10, 12.5, 15, 20, and 30% remote weights. For each point,
   report median decode tok/s, prompt tok/s, H2D GB/s, GPU DRAM GB/s, power, and
   output/logit agreement.
5. Run three controls: all-VRAM on the 32 GiB card, llama.cpp's normal CPU
   offload under a 16 GiB fit target, and the new striped path under the same
   16 GiB target.
6. Use 4K, 8K, and 16K contexts. Qwen3.5-27B Q4_K_M is about 15.95 GiB, while
   its 16 full-attention layers use about 64 KiB of F16 KV per token; the other
   48 layers have a roughly 150 MiB fixed recurrent state.

The decisive Qwen3.5 run needs the actual Q4_K_M file, which is not currently on
the machine. The existing Gemma model is sufficient for the loader/capacity and
traditional-offload controls, but not for validating the Qwen3.5 hybrid cache.

An unrelated layer-streaming prototype was also inspected. It streams whole
residency units through SSD/VRAM rather than overlapping RAM and VRAM weight
traffic, so it is not the striping implementation in this repository.

## Build and run

Apply and build the final llama.cpp implementation:

```bash
git checkout b820cc8
git apply /path/to/intertidal-vmm.patch
cmake -S . -B build-intertidal -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-intertidal --target llama-bench -j

# Keep the same 2 MiB page budget but switch the remote backing transport.
# staging is the default; zero_copy directly maps pinned host-NUMA pages at
# the original tensor virtual addresses and performs no H2D prefetch/staging.
export GGML_CUDA_HYBRID_MODE=staging
# export GGML_CUDA_HYBRID_MODE=zero_copy
export GGML_CUDA_HYBRID_PAGE_BUDGET=266

PYTHONPATH=$PWD/gguf-py env -u LD_PRELOAD python3 gemma_vmm_bench.py \
  --model /path/to/gemma-4-31B-it-Q4_0.gguf \
  --bench $PWD/build-intertidal/bin/llama-bench \
  --output vmm-final.json \
  --pages 266 --prompt 512 --generation 64 --ubatch 512 \
  --repetitions 10 --cases original,local,vmm
```

The original standalone memory-bandwidth microbenchmark can still be built as
follows:

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120a \
  -o striping_bench striping_bench.cu

./striping_bench
./striping_bench --write-combined
./striping_bench --staged
```

## Fixed-model capacity sweep

`gemma_capacity_sweep.py` is the resumable local orchestrator for the full
Gemma-4 experiment.  It keeps the model fixed, reserves KV capacity with
`--ctx-size`, and compares VMM DMA staging, VMM CUDA host zero-copy, and native
whole-layer `-ngl` CPU offload.

The plan has two deliberately separate phases:

- `pure_curve` runs one complete 1.0-percentage-point pilot followed by the
  0.1-point dense curve at the 31 GiB working set, stopping after two
  consecutive points at or below 10% of the initial prefill or decode speed;
- `capacity_frontier` visits the 31.0--34.0 GiB KV working-set grid and starts
  at the theoretical gross-offload lower bound.  It advances in 0.1-point
  steps through OOM or insufficient-net-saving cases, then records the first
  fit and one following confirmation point.  An all-local control at the exact
  context records the expected OOM before an over-capacity search jumps to its
  rigorous lower bound.  Native offload analogously finds the smallest number
  of CPU layers rather than rerunning a full curve.  A successful native run
  counts as a frontier only when its process-peak reduction covers the required
  capacity: the reduction uses a measured same-context `-ngl 61` peak when it
  fits, or an explicitly labelled calibrated/inferred all-local peak when the
  all-local allocation is above physical VRAM.  Native CPU buffer size is
  retained as loader diagnostics and is never reported as gross remote bytes.

Copy and edit `capacity-sweep-manifest.example.json`, then inspect the complete
adaptive plan without opening SSH:

```bash
python3 gemma_capacity_sweep.py \
  --manifest capacity-sweep-manifest.json \
  --output results/gemma-capacity-sweep.csv \
  --dry-run
```

Remove `--dry-run` to execute.  Each benchmark's stdout/stderr is streamed
directly over SSH, parsed locally, and appended to the resumable CSV together
with OOM status, realized page/net-byte accounting, process/device peak VRAM,
graphics clocks, temperature, power, and utilization.  The wrapper creates no
remote result or log file.  Password-only SSH should first establish a local
ControlMaster connection (or use an SSH key), because wrapper stdin carries
the script itself.

Correctness is a mandatory per-case gate by default.  The benchmark emits one
ordered checksum and a non-finite-logit count after every warmup and measured
prompt/decode pass.  With five repetitions and warmup enabled the required
sequence is six `prompt` entries followed by six `decode` entries.  The first
successful 0% all-local case supplies the baseline automatically; alternatively
set `correctness.baseline_checksums.prompt` and `.decode` explicitly in the
manifest.  Every Intertidal/zero-copy case must match the complete sequence,
not only the final checksum.  A missing line, wrong count/order, checksum
mismatch, NaN/Inf, or unavailable baseline records `correctness_fail` and then
stops the sweep immediately.  Native `-ngl` logits can differ numerically by
backend, so that path enforces sequence completeness and finiteness unless it
is explicitly added to `strict_checksum_schemes`.

Render the append-only CSV while the sweep is still in progress or after it
finishes:

```bash
python3 plot_capacity_results.py \
  --input results/gemma-capacity-sweep.csv \
  --output results/gemma-capacity-performance
```

This writes both SVG and PNG.  The four panels show prefill and decode
retention against measured logical working set and against realized gross RAM
weight percentage.  The 95% acceptance target and 10% stop threshold are
drawn directly on every panel.  Duplicate measurements of the same case gain
a 95% Student-t confidence interval; otherwise the chart stays a measured
point/line plot.  Native whole-layer CPU offload is markers-only, so its sparse
residency choices are never presented as an interpolated 0.1% curve.  The
hybrid pure-curve x-axis uses `gross_remote_pp`; because native offload has no
VMM gross-page concept, its marker x-coordinate is the loader-reported
`CPU / (CPU + CUDA)` model-buffer fraction.

For PCIe/DMA profiling rather than the capacity sweep, follow
[`PROFILING.md`](PROFILING.md).  It defines the matched H2D calibration,
profile-off/direct-control/deep observer trio, selected prefill/decode probes,
and the local `analyze_profile.py` CSV/Markdown/SVG report.
