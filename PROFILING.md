# Intertidal profiling runbook

This runbook is for diagnosing the VMM staging and host-NUMA zero-copy paths
without confusing profiler overhead with the performance of the production
CUDA Graph path.  The profiling records are diagnostic evidence; the ordinary
profile-off benchmark remains the performance result.

## What is measured

The CUDA backend emits an opt-in, versioned `hybrid_profile` protocol.  In
staging mode, a sampled remote-layer transfer is split into:

```text
main stream kick -> copy-stream queue -> H2D DMA -> ready
                                      \-> exposed main-stream wait -> layer work
```

The final records contain totals and alternate per-evaluation/per-layer views:

- actual copy count and bytes, independent of the timing sample rate;
- sampled copy, kick-to-ready, queue, exposed-wait, hidden, and work time;
- effective H2D GB/s, prefetch hit rate, overlap percentage, and dropped samples;
- the eight layers with the largest accumulated exposed wait;
- zero-copy remote touches and estimated remote bytes;
- cross-evaluation wrap-prefetch attribution.

The runner writes one append-only local JSONL sidecar record for each executed
case.  It also samples process/device VRAM, GPU and memory clocks, temperature,
power, compute/memory utilization, PCIe RX/TX when the driver exposes it,
process RSS/locked memory/CPU/I/O/major faults, host available memory/swap, and
CPU/memory/I/O pressure.  No result file is created on the GPU host.
Deep mode enables the driver's one-second `nvidia-smi dmon -s t` PCIe fallback;
if this driver rejects that counter group, it remains null without failing the
case.  Use `--profile-no-pcie-dmon` when measuring a very short observer-control
case where even that background process is undesirable.

## Bandwidth and overlap definitions

Use decimal GB/s consistently:

```text
B_dma                 = sampled H2D bytes / CUDA-event copy time
PCIe single-copy util = B_dma / matched-size pinned-H2D single-copy ceiling
PCIe sustained util   = B_dma / matched-size pinned-H2D copy-train ceiling
exposed stall         = main-stream wait measured around the dependency event
overlap               = 1 - exposed stall / kick-to-ready time
```

For heterogeneous layer sizes, the analyzer first chooses a calibrated ceiling
`B_i` for each sampled layer and combines them as
`B_mix = sum(bytes_i) / sum(bytes_i / B_i)`.  This avoids treating a mixture of
2 MiB and large layer transfers as one fictitious average-sized copy.

The single-copy ceiling is the primary denominator because it matches the
per-layer CUDA-event measurement.  The sustained denominator shows how close
the implementation is to the link's copy-train ceiling.  Both ceilings must be
measured on the same boot, GPU power/clock policy, NUMA placement, host
allocation type, and relevant transfer sizes.  `nvidia-smi` PCIe counters are
a coarse cross-check, not the denominator.

`utilization.memory` is not GPU DRAM bandwidth.  GDDR/L2/SM throughput requires
a selected-kernel Nsight Compute pass after the low-overhead run identifies the
important kernel and layer.

## Required observer controls

Run every diagnostic point in this order:

1. `light`: normal CUDA Graph path, system counters only.  This is the closest
   profiled result to production.
2. `deep --profile-backend-off`: the same 0.2-second system sampler, PCIe dmon,
   and direct execution as deep mode, but no CUDA timing events.  This is the
   matched observer control.
3. `deep`: direct execution plus CUDA timing events and detailed backend
   records.  The difference from step 2 is profiler-event/logging overhead.

Never normalize the production curve to step 3.  Deep mode forces direct
execution because timing event semantics inside capture/replay would otherwise
be ambiguous.

## Calibrate pinned H2D first

Compile the calibration source in a temporary in-memory directory on the test
host.  The source and binary must be streamed from/to the local workstation;
do not leave them on the 5090 filesystem.

```bash
ssh wici@192.168.1.182 \
  'bin=/dev/shm/pcie-h2d-calibrate.$$; \
   trap '\''rm -f "$bin"'\'' EXIT INT TERM HUP; \
   nvcc -O3 -std=c++17 -x cu - -o "$bin" && \
   "$bin" --device 0 --warmup 20 --repetitions 100 \
     --sizes-mib 2,4,8,16,32,64,128,256,512' \
  < pcie_h2d_calibrate.cu \
  > results/pcie-h2d-calibration.jsonl
```

The calibrator uses a non-blocking stream and portable write-combined pinned
RAM, matching the staging source allocation.  It reports per-size single-copy
p50/p95/p99 latency and a long copy-train ceiling.  After the first deep run,
repeat calibration with any exact layer sizes that fall far between defaults.

## Selected Q4 probes

Use a fresh CSV for profiling so resume logic cannot skip cases already present
in the main 0.1-point sweep.  Each command prepends a 0% checksum control.  The
first priority is the decode-defined 90--95% sweet spot.  Historical locator
data places its boundaries around 3.3--3.9% gross offload; reconfirm these with
the current binary before optimizing.  Keep 10% and the current high-offload
region as later full-curve diagnosis points.

Before profiling, deploy the current patch to a fresh temporary build and pass
its executable/library path in the manifest.  Do not reuse the old temporary
library directory shown in the saved RTX manifest unless it still exists and
its SHA matches this repository.  The runner itself streams stdout/stderr home
and creates no remote result file; remove the temporary build after the last
case.

```bash
cd /Users/hachima/pcie-striping-experiment

python3 -u gemma_capacity_sweep.py \
  --manifest capacity-sweep-manifest.rtx5090.json \
  --output results/profile-q4-light.csv \
  --schemes intertidal_dma \
  --only-offload-pp 3.3,3.4,3.6,3.7,3.8,3.9 \
  --profile-benchmark prefill \
  --profile light \
  --profile-output results/profile-q4-light.jsonl

python3 -u gemma_capacity_sweep.py \
  --manifest capacity-sweep-manifest.rtx5090.json \
  --output results/profile-q4-direct-control.csv \
  --schemes intertidal_dma \
  --only-offload-pp 3.3,3.4,3.6,3.7,3.8,3.9 \
  --profile-benchmark prefill \
  --profile deep --profile-backend-off \
  --profile-output results/profile-q4-direct-control.jsonl

python3 -u gemma_capacity_sweep.py \
  --manifest capacity-sweep-manifest.rtx5090.json \
  --output results/profile-q4-deep.csv \
  --schemes intertidal_dma \
  --only-offload-pp 3.3,3.4,3.6,3.7,3.8,3.9 \
  --profile-benchmark prefill \
  --profile deep --profile-raw \
  --profile-output results/profile-q4-deep.jsonl
```

Repeat all three commands with `--profile-benchmark decode` and distinct output
filenames.  This keeps prefill and decode backend timing in separate processes,
so a slow layer/eval cannot be assigned to the wrong workload.  Then repeat the
selected deep command with `--schemes cuda_zero_copy`.  Zero-copy
has no explicit DMA event to time, so use PCIe RX plus the per-eval work window
for the first diagnosis, then confirm system-memory request latency and SM/L2
behavior with Nsight tools.

The Q8 capacity runner accepts the same `--profile`, `--profile-output`, and
`--profile-raw` controls.  Profile only a few frontier contexts after the
minimum page budget has been established; do not run deep instrumentation over
the entire binary-search or 0.1-point sweep.

## Reading the report

Run the local analyzer against the deep JSONL and same-session calibration:

```bash
python3 analyze_profile.py \
  results/profile-q4-deep.jsonl \
  --calibration-jsonl results/pcie-h2d-calibration.jsonl \
  --output-prefix results/profile-q4
```

It writes a flat CSV, Markdown diagnosis, and SVG overview.  If calibration is
temporarily unavailable, pass `--h2d-ceiling-gbps 48.73`; this is less rigorous
because it does not match transfer size.

Use these as heuristics, not hard pass/fail rules:

| Signal | Likely bottleneck | First experiment |
| --- | --- | --- |
| High PCIe utilization, high exposed wait | Link bandwidth is exhausted | Fewer remote bytes or more compute per staged byte |
| Low utilization, high copy p95, small transfers | Transfer fragmentation/latency | Coalesce layer pages or increase copy size |
| Low utilization, high copy-stream queue | Copy-stream serialization or host pressure | Inspect prefetch distance, pinning and NUMA locality |
| Good DMA GB/s, low overlap | Prefetch starts too late | Move the layer trigger earlier or add safe buffering |
| A few layers dominate wait | Layout/parity/schedule imbalance | Redistribute page budget or specialize those layers |
| Major faults, memory PSI, falling locked RAM | Host-memory path is unstable | Fix pinning, memory headroom and NUMA placement |
| Falling clocks/P-state/power ceiling | Thermal or power throttling | Stabilize fan/power/clock policy before comparing |
| Zero-copy PCIe traffic but long work windows | System-memory load latency/low ILP | Inspect selected kernels with Nsight Compute |

## Deep NVIDIA traces

Only run these after the JSONL report identifies one representative case.  Put
temporary traces in tmpfs, copy them to the local private repository, then
delete the remote copy.  Nsight Systems should trace CUDA and NVTX/OS runtime
for a short prompt/decode window; disable CPU sampling to reduce disturbance:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /dev/shm/intertidal-one-case \
  <the exact direct-control benchmark command>
```

Inspect stream overlap, `cudaMemcpyAsync`, event waits, MMQ/MMVQ kernels, and
gaps between submissions.  Then use Nsight Compute on one kernel occurrence
selected from that trace, starting with the basic/SpeedOfLight and memory
workload sections rather than profiling the whole model.  Record the exact
kernel regex and launch skip/count beside the report so it is reproducible.

## Acceptance gates

- Deep and direct-control cases must pass the same finite/checksum gate as the
  production path.
- `backend.dropped` must be zero; otherwise enlarge the timing ring or sample
  less frequently and repeat.
- Q4 selected probes use all transfers with a 256-sample ring.  The Q8
  long-context manifest samples every 16th transfer with a 1024-sample ring;
  `timing_is_sampled` stays explicit and utilization uses sampled bytes only.
- Keep PCIe link at Gen5 x16 and compare clocks, temperature and power across
  observer controls.
- At least two warm-up evaluations and five measured repeats are needed for
  selected short cases; report median and spread.
- A useful optimization must improve profile-off throughput.  A prettier deep
  trace alone is not a performance result.
