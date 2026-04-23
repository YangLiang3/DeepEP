#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kernels/api.cuh"

namespace deep_ep {
namespace internode_runtime {

enum class RuntimeBackend {
    CUDA_NVSHMEM,
};

inline RuntimeBackend selected_backend() {
    return RuntimeBackend::CUDA_NVSHMEM;
}

inline const char* selected_backend_name() {
    return "cuda_nvshmem";
}

inline std::vector<uint8_t> get_unique_id() {
    return internode::get_unique_id();
}

inline int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    return internode::init(root_unique_id_val, rank, num_ranks, low_latency_mode);
}

inline void* alloc(size_t size, size_t alignment) {
    return internode::alloc(size, alignment);
}

inline void free(void* ptr) {
    internode::free(ptr);
}

inline void barrier() {
    internode::barrier();
}

inline void finalize() {
    internode::finalize();
}

}  // namespace internode_runtime
}  // namespace deep_ep
