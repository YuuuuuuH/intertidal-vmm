#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1024ull * 1024ull;
constexpr std::size_t kChunkBytes = kMiB;
constexpr int kTotalChunks = 8000;
constexpr int kGroupChunks = 100;
constexpr int kThreads = 256;
constexpr int kMaxRemotePercent = 40;
constexpr int kWarmupRuns = 2;
constexpr int kMeasuredRuns = 10;
constexpr int kStageChunks = 128;

static_assert(kTotalChunks % kGroupChunks == 0);
static_assert(kChunkBytes % sizeof(uint4) == 0);

void check_cuda(cudaError_t status, const char * expression, const char * file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "CUDA error at %s:%d: %s: %s\n", file, line, expression,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

__global__ void striped_scan(const uint4 * local, const uint4 * remote,
                             std::uint32_t * checksums, int remote_chunks_per_group) {
    const int global_chunk = blockIdx.x;
    const int group = global_chunk / kGroupChunks;
    const int position = global_chunk % kGroupChunks;
    const bool is_remote = position < remote_chunks_per_group;

    const int source_chunk = is_remote
        ? group * remote_chunks_per_group + position
        : group * (kGroupChunks - remote_chunks_per_group) +
              (position - remote_chunks_per_group);
    const uint4 * source = (is_remote ? remote : local) +
                           static_cast<std::size_t>(source_chunk) *
                               (kChunkBytes / sizeof(uint4));

    std::uint32_t accumulator0 = static_cast<std::uint32_t>(global_chunk + threadIdx.x);
    std::uint32_t accumulator1 = accumulator0 + 1;
    std::uint32_t accumulator2 = accumulator0 + 2;
    std::uint32_t accumulator3 = accumulator0 + 3;
    for (std::size_t i = threadIdx.x;
         i < kChunkBytes / sizeof(uint4); i += static_cast<std::size_t>(blockDim.x) * 4) {
        const uint4 value0 = source[i + 0];
        const uint4 value1 = source[i + blockDim.x];
        const uint4 value2 = source[i + 2 * blockDim.x];
        const uint4 value3 = source[i + 3 * blockDim.x];
        accumulator0 += value0.x + value0.y + value0.z + value0.w;
        accumulator1 += value1.x + value1.y + value1.z + value1.w;
        accumulator2 += value2.x + value2.y + value2.z + value2.w;
        accumulator3 += value3.x + value3.y + value3.z + value3.w;
    }

    __shared__ std::uint32_t partial[kThreads];
    partial[threadIdx.x] = accumulator0 + accumulator1 + accumulator2 + accumulator3;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        checksums[global_chunk] = partial[0];
    }
}

__global__ void contiguous_scan(const uint4 * source, std::uint32_t * checksums) {
    const int source_chunk = blockIdx.x;
    source += static_cast<std::size_t>(source_chunk) * (kChunkBytes / sizeof(uint4));

    std::uint32_t accumulator0 = static_cast<std::uint32_t>(source_chunk + threadIdx.x);
    std::uint32_t accumulator1 = accumulator0 + 1;
    std::uint32_t accumulator2 = accumulator0 + 2;
    std::uint32_t accumulator3 = accumulator0 + 3;
    for (std::size_t i = threadIdx.x;
         i < kChunkBytes / sizeof(uint4); i += static_cast<std::size_t>(blockDim.x) * 4) {
        const uint4 value0 = source[i + 0];
        const uint4 value1 = source[i + blockDim.x];
        const uint4 value2 = source[i + 2 * blockDim.x];
        const uint4 value3 = source[i + 3 * blockDim.x];
        accumulator0 += value0.x + value0.y + value0.z + value0.w;
        accumulator1 += value1.x + value1.y + value1.z + value1.w;
        accumulator2 += value2.x + value2.y + value2.z + value2.w;
        accumulator3 += value3.x + value3.y + value3.z + value3.w;
    }

    __shared__ std::uint32_t partial[kThreads];
    partial[threadIdx.x] = accumulator0 + accumulator1 + accumulator2 + accumulator3;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        checksums[source_chunk] = partial[0];
    }
}

double run_case(const uint4 * local, const uint4 * remote, std::uint32_t * checksums,
                int remote_percent) {
    const int remote_chunks_per_group = remote_percent;
    for (int i = 0; i < kWarmupRuns; ++i) {
        striped_scan<<<kTotalChunks, kThreads>>>(local, remote, checksums,
                                                remote_chunks_per_group);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kMeasuredRuns; ++i) {
        striped_scan<<<kTotalChunks, kThreads>>>(local, remote, checksums,
                                                remote_chunks_per_group);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed_ms) / kMeasuredRuns;
}

double run_staged_case(const uint4 * local, const uint4 * remote_host,
                       std::uint32_t * checksums, int remote_percent) {
    const int remote_chunks = kTotalChunks * remote_percent / 100;
    const int local_chunks = kTotalChunks - remote_chunks;

    cudaStream_t local_stream;
    cudaStream_t remote_streams[2];
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEvent_t done[3];
    uint4 * staging[2]{};

    CUDA_CHECK(cudaStreamCreateWithFlags(&local_stream, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&remote_streams[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&staging[i], static_cast<std::size_t>(kStageChunks) * kChunkBytes));
    }
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (auto & event : done) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    double measured_ms = 0.0;
    for (int run = -kWarmupRuns; run < kMeasuredRuns; ++run) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaStreamWaitEvent(local_stream, start));
        for (auto & stream : remote_streams) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, start));
        }

        contiguous_scan<<<local_chunks, kThreads, 0, local_stream>>>(local, checksums);
        for (int offset = 0, tile = 0; offset < remote_chunks;
             offset += kStageChunks, ++tile) {
            const int chunks = std::min(kStageChunks, remote_chunks - offset);
            const int stream_index = tile % 2;
            const std::size_t bytes = static_cast<std::size_t>(chunks) * kChunkBytes;
            CUDA_CHECK(cudaMemcpyAsync(staging[stream_index],
                                       remote_host +
                                           static_cast<std::size_t>(offset) *
                                               (kChunkBytes / sizeof(uint4)),
                                       bytes, cudaMemcpyHostToDevice,
                                       remote_streams[stream_index]));
            contiguous_scan<<<chunks, kThreads, 0, remote_streams[stream_index]>>>(
                staging[stream_index], checksums + local_chunks + offset);
        }

        CUDA_CHECK(cudaEventRecord(done[0], local_stream));
        CUDA_CHECK(cudaEventRecord(done[1], remote_streams[0]));
        CUDA_CHECK(cudaEventRecord(done[2], remote_streams[1]));
        for (auto & event : done) {
            CUDA_CHECK(cudaStreamWaitEvent(nullptr, event));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        if (run >= 0) {
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            measured_ms += elapsed_ms;
        }
    }

    for (auto & event : done) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaFree(staging[i]));
        CUDA_CHECK(cudaStreamDestroy(remote_streams[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(local_stream));
    return measured_ms / kMeasuredRuns;
}

}  // namespace

int main(int argc, char ** argv) {
    bool write_combined = false;
    bool staged = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--write-combined") == 0) {
            write_combined = true;
        } else if (std::strcmp(argv[i], "--staged") == 0) {
            staged = true;
            write_combined = true;
        } else {
            std::fprintf(stderr, "usage: %s [--write-combined] [--staged]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (!properties.canMapHostMemory) {
        std::fprintf(stderr, "Device does not support mapped host memory.\n");
        return EXIT_FAILURE;
    }

    const std::size_t total_bytes = static_cast<std::size_t>(kTotalChunks) * kChunkBytes;
    const std::size_t max_remote_bytes = total_bytes * kMaxRemotePercent / 100;

    uint4 * local = nullptr;
    void * remote_host = nullptr;
    uint4 * remote_device = nullptr;
    std::uint32_t * checksums = nullptr;

    CUDA_CHECK(cudaMalloc(&local, total_bytes));
    CUDA_CHECK(cudaMemset(local, 1, total_bytes));
    unsigned int host_flags = cudaHostAllocMapped | cudaHostAllocPortable;
    if (write_combined) {
        host_flags |= cudaHostAllocWriteCombined;
    }
    CUDA_CHECK(cudaHostAlloc(&remote_host, max_remote_bytes, host_flags));
    std::memset(remote_host, 1, max_remote_bytes);
    CUDA_CHECK(cudaHostGetDevicePointer(&remote_device, remote_host, 0));
    CUDA_CHECK(cudaMalloc(&checksums,
                          static_cast<std::size_t>(kTotalChunks) * sizeof(std::uint32_t)));

    const std::vector<int> percentages{0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40};
    std::vector<double> times_ms;
    times_ms.reserve(percentages.size());
    for (const int percentage : percentages) {
        times_ms.push_back(staged
                               ? run_staged_case(local, static_cast<const uint4 *>(remote_host),
                                                 checksums, percentage)
                               : run_case(local, remote_device, checksums, percentage));
    }

    const double local_ms = times_ms.front();
    std::printf("device,%s\n", properties.name);
    std::printf("mode,%s\n", staged ? "dma_staged" : "zero_copy");
    std::printf("host_allocation,%s\n", write_combined ? "write_combined" : "default");
    std::printf("total_bytes,%zu\n", total_bytes);
    std::printf("remote_pct,local_GiB,remote_GiB,time_ms,aggregate_GBps,relative_to_local\n");
    for (std::size_t i = 0; i < percentages.size(); ++i) {
        const int percentage = percentages[i];
        const double remote_bytes = static_cast<double>(total_bytes) * percentage / 100.0;
        const double local_bytes = static_cast<double>(total_bytes) - remote_bytes;
        const double seconds = times_ms[i] / 1000.0;
        const double gbps = static_cast<double>(total_bytes) / seconds / 1.0e9;
        const double relative = local_ms / times_ms[i];
        std::printf("%d,%.3f,%.3f,%.3f,%.2f,%.4f\n", percentage,
                    local_bytes / (1024.0 * 1024.0 * 1024.0),
                    remote_bytes / (1024.0 * 1024.0 * 1024.0), times_ms[i], gbps,
                    relative);
    }

    CUDA_CHECK(cudaFree(checksums));
    CUDA_CHECK(cudaFreeHost(remote_host));
    CUDA_CHECK(cudaFree(local));
    return EXIT_SUCCESS;
}
