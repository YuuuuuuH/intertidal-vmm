#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1024ull * 1024ull;

void check_cuda(cudaError_t status, const char * expression, const char * file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "CUDA error at %s:%d: %s: %s\n", file, line, expression,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

struct options {
    int device = 0;
    int warmup = 20;
    int repetitions = 100;
    std::vector<std::size_t> sizes = {
        2 * kMiB, 4 * kMiB, 8 * kMiB, 16 * kMiB,
        32 * kMiB, 64 * kMiB, 128 * kMiB, 256 * kMiB, 512 * kMiB,
    };
};

[[noreturn]] void usage(const char * argv0, const char * error = nullptr) {
    if (error != nullptr) {
        std::fprintf(stderr, "error: %s\n", error);
    }
    std::fprintf(stderr,
        "usage: %s [--device N] [--warmup N] [--repetitions N] "
        "[--sizes-mib 2,4,8,16,32,64,128,256,512]\n",
        argv0);
    std::exit(error == nullptr ? EXIT_SUCCESS : EXIT_FAILURE);
}

int parse_positive_int(const char * value, const char * option, bool allow_zero = false) {
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < (allow_zero ? 0 : 1) ||
        parsed > std::numeric_limits<int>::max()) {
        std::string message = std::string("invalid value for ") + option + ": " + value;
        usage("pcie_h2d_calibrate", message.c_str());
    }
    return static_cast<int>(parsed);
}

std::vector<std::size_t> parse_sizes(const char * value) {
    std::vector<std::size_t> result;
    const std::string text(value);
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::string field = text.substr(begin, comma - begin);
        if (field.empty()) {
            usage("pcie_h2d_calibrate", "--sizes-mib contains an empty value");
        }
        errno = 0;
        char * end = nullptr;
        const double mib = std::strtod(field.c_str(), &end);
        if (errno != 0 || end == field.c_str() || *end != '\0' || !std::isfinite(mib) || mib <= 0.0) {
            usage("pcie_h2d_calibrate", "--sizes-mib values must be positive numbers");
        }
        const double bytes = mib * static_cast<double>(kMiB);
        if (bytes > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
            usage("pcie_h2d_calibrate", "--sizes-mib value is too large");
        }
        result.push_back(static_cast<std::size_t>(std::llround(bytes)));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

options parse_options(int argc, char ** argv) {
    options parsed;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        auto require_value = [&]() -> const char * {
            if (++i >= argc) {
                usage(argv[0], (argument + " requires a value").c_str());
            }
            return argv[i];
        };
        if (argument == "--device") {
            parsed.device = parse_positive_int(require_value(), "--device", true);
        } else if (argument == "--warmup") {
            parsed.warmup = parse_positive_int(require_value(), "--warmup", true);
        } else if (argument == "--repetitions") {
            parsed.repetitions = parse_positive_int(require_value(), "--repetitions");
        } else if (argument == "--sizes-mib") {
            parsed.sizes = parse_sizes(require_value());
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
        } else {
            usage(argv[0], ("unknown option: " + argument).c_str());
        }
    }
    return parsed;
}

double percentile(std::vector<float> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return static_cast<double>(values[lower]) * (1.0 - weight) +
           static_cast<double>(values[upper]) * weight;
}

double gbps(std::size_t bytes, double milliseconds) {
    return milliseconds > 0.0 ? static_cast<double>(bytes) / milliseconds / 1.0e6 : 0.0;
}

}  // namespace

int main(int argc, char ** argv) {
    const options opts = parse_options(argc, argv);
    CUDA_CHECK(cudaSetDevice(opts.device));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, opts.device));
    const std::size_t maximum = *std::max_element(opts.sizes.begin(), opts.sizes.end());

    void * host = nullptr;
    void * device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, maximum, cudaHostAllocPortable | cudaHostAllocWriteCombined));
    std::memset(host, 0xa5, maximum);
    CUDA_CHECK(cudaMalloc(&device, maximum));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::printf(
        "{\"protocol\":\"intertidal-pcie-calibration-v1\",\"event\":\"config\","
        "\"device\":%d,\"device_name\":\"%s\",\"warmup\":%d,\"repetitions\":%d,"
        "\"host_allocation\":\"portable_write_combined\",\"stream\":\"non_blocking\"}\n",
        opts.device, properties.name, opts.warmup, opts.repetitions);

    for (const std::size_t bytes : opts.sizes) {
        for (int i = 0; i < opts.warmup; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> individual_ms;
        individual_ms.reserve(static_cast<std::size_t>(opts.repetitions));
        for (int i = 0; i < opts.repetitions; ++i) {
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUDA_CHECK(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
            individual_ms.push_back(elapsed);
        }

        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < opts.repetitions; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float train_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&train_ms, start, stop));

        const double p50_ms = percentile(individual_ms, 0.50);
        const double p95_ms = percentile(individual_ms, 0.95);
        const double p99_ms = percentile(individual_ms, 0.99);
        const double sustained_ms = static_cast<double>(train_ms) / opts.repetitions;
        std::printf(
            "{\"protocol\":\"intertidal-pcie-calibration-v1\",\"event\":\"result\","
            "\"direction\":\"h2d\",\"bytes\":%zu,\"mib\":%.6f,\"repetitions\":%d,"
            "\"single_p50_ms\":%.9f,\"single_p95_ms\":%.9f,\"single_p99_ms\":%.9f,"
            "\"single_p50_gbps_decimal\":%.6f,\"sustained_ms_per_copy\":%.9f,"
            "\"sustained_gbps_decimal\":%.6f}\n",
            bytes, static_cast<double>(bytes) / kMiB, opts.repetitions,
            p50_ms, p95_ms, p99_ms, gbps(bytes, p50_ms), sustained_ms,
            gbps(bytes, sustained_ms));
        std::fflush(stdout);
    }

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device));
    CUDA_CHECK(cudaFreeHost(host));
    return EXIT_SUCCESS;
}
