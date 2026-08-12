#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#define CU_OK(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) { \
    const char * s_ = nullptr; cuGetErrorString(r_, &s_); \
    std::fprintf(stderr, "%s failed: %s\n", #call, s_ ? s_ : "unknown"); return 1; } } while (0)
#define CUDA_OK(call) do { cudaError_t r_ = (call); if (r_ != cudaSuccess) { \
    std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(r_)); return 1; } } while (0)

int main() {
    CU_OK(cuInit(0));
    CUdevice dev;
    CUcontext ctx;
    CU_OK(cuDeviceGet(&dev, 0));
    CUDA_OK(cudaFree(nullptr));
    CU_OK(cuCtxGetCurrent(&ctx));

    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    size_t minimum = 0, recommended = 0;
    CU_OK(cuMemGetAllocationGranularity(&minimum, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CU_OK(cuMemGetAllocationGranularity(&recommended, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    std::printf("minimum=%zu recommended=%zu\n", minimum, recommended);

    const size_t bytes = minimum * 4;
    CUmemGenericAllocationHandle handle;
    CU_OK(cuMemCreate(&handle, bytes, &prop, 0));
    CUdeviceptr va0 = 0, va1 = 0;
    CU_OK(cuMemAddressReserve(&va0, bytes, minimum, 0, 0));
    CU_OK(cuMemAddressReserve(&va1, bytes, minimum, 0, 0));
    CU_OK(cuMemMap(va0, bytes, 0, handle, 0));
    CU_OK(cuMemMap(va1, bytes, 0, handle, 0));
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_OK(cuMemSetAccess(va0, bytes, &access, 1));
    CU_OK(cuMemSetAccess(va1, bytes, &access, 1));

    CUDA_OK(cudaMemset(reinterpret_cast<void *>(va0), 0x5a, bytes));
    std::vector<unsigned char> host(64);
    CUDA_OK(cudaMemcpy(host.data(), reinterpret_cast<void *>(va1), host.size(), cudaMemcpyDeviceToHost));
    for (unsigned char value : host) {
        if (value != 0x5a) {
            std::fprintf(stderr, "alias mismatch\n");
            return 2;
        }
    }
    std::puts("alias_ok");
    CU_OK(cuMemUnmap(va0, bytes));
    CU_OK(cuMemUnmap(va1, bytes));
    CU_OK(cuMemAddressFree(va0, bytes));
    CU_OK(cuMemAddressFree(va1, bytes));
    CU_OK(cuMemRelease(handle));
    return 0;
}
