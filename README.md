# PCIe weight striping experiment

Measured on the RTX 5090 test host on 2026-08-11 and 2026-08-12.

## Final result: VMM double-buffer staging

The implementation target has been met on the RTX 5090 with Gemma-4-31B
Q4_0.  The final path is not the early row-split prototype described later in
this document.  It uses CUDA VMM to give every weight tensor a stable,
contiguous virtual address:

- ordinary pages map to private VRAM;
- selected complete 2 MiB pages alias one of two physical staging slots;
- remote pages are packed by transformer layer in pinned RAM;
- layer `L+1` is copied on a non-blocking stream while layer `L` computes;
- the last layer ring-prefetches layer zero for the next CUDA Graph launch;
- original MMQ/MMVQ, stream-K, fusions, and CUDA Graphs remain enabled.

There is no special remote matrix kernel and no output merge.  The original
kernel dereferences the original tensor pointer after the staging physical
pages have been filled.

Final strict capacity point, 10 repetitions, `pp512`, `tg64`, `ubatch=512`,
Flash Attention on:

| Case | Prefill tok/s | Decode tok/s | Retention | Net VRAM released |
| --- | ---: | ---: | ---: | ---: |
| Original all-VRAM (post control) | 2262.12 | 75.48 | 100% | 0 |
| VMM + H2D, 266 pages | 2190.11 | 73.00 | 96.82% / 96.72% | 513.083 MiB |

The staged layout was 532 MiB of remote weight pages across all 60 layers and
only 18 MiB of physical staging VRAM.  Including VMM end-page rounding, the
measured net saving was 513.083 MiB out of a 16.133 GiB model, or 3.1058% of
the logical weight footprint.  The corresponding capacity multiplier is
`1 / (1 - 0.031058) = 1.03205`, so a 32 GiB local budget becomes 33.026 GiB.
This is strictly above the 32-to-33 GiB target while both prefill and decode
remain above 95%.

The all-local VMM control reached 76.27 decode tok/s versus 75.48 for the
original allocator, showing that stable virtual addressing and the original
compute kernels add no measurable decode tax.  Prefill on this host has large
run-to-run thermal/clock variance; the reported H2D point uses the later,
higher original baseline as the conservative denominator.

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
- `gemma_vmm_bench.py`: balanced page selector and three-case runner.
- `vmm-ring2-final-266-r10.json`: final 266-page measurements.
- `vmm-final-original-post-r10.json`: post-run all-VRAM control.
- `vmm-final-local-r10.json`: all-local VMM control.
- `vmm-hybrid-266.err`: layout and final correctness checksums.
- `intertidal-audit.txt`: independent VRAM, CUDA Graph, and 64-token audit.

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

The useful equations are:

```text
optimal remote fraction of total weights = B_pcie / (B_gpu + B_pcie)
no-loss effective capacity               = V_local * (1 + B_pcie / B_gpu)
remote-bound relative throughput         = B_pcie / (remote_fraction * B_gpu)
```

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
