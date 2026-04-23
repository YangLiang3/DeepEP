#pragma once

// SYCL port of kernels/launch.cuh
//
// Provides SYCL nd_range launch configuration helpers and dispatch macros.
//
// CUDA launch model:
//   kernel<<<num_blocks, threads_per_block, shared_mem, stream>>>(args...)
//
// SYCL launch model:
//   queue.submit([&](sycl::handler& cgh) {
//       cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) { ... });
//   });

#include <sycl/sycl.hpp>
#include "sycl_backend/configs.hpp"
#include "sycl_backend/exception.hpp"
#include "sycl_backend/sycl_utils.hpp"

namespace deep_ep {

// ---------------------------------------------------------------------------
// SYCL launch configuration
// ---------------------------------------------------------------------------
struct SyclLaunchConfig {
    int num_work_groups;   // analogous to CUDA grid.x (num_sms)
    int work_group_size;   // analogous to CUDA block.x (threads per block)
    int local_mem_bytes;   // dynamic local (shared) memory

    SyclLaunchConfig(int num_wgs, int wg_size, int local_bytes = 0)
        : num_work_groups(num_wgs), work_group_size(wg_size), local_mem_bytes(local_bytes) {}

    sycl::nd_range<1> nd_range() const {
        return sycl::nd_range<1>(
            sycl::range<1>(num_work_groups * work_group_size),
            sycl::range<1>(work_group_size));
    }
};

// ---------------------------------------------------------------------------
// SETUP_SYCL_LAUNCH_CONFIG — create a SyclLaunchConfig
// ---------------------------------------------------------------------------
#define SETUP_SYCL_LAUNCH_CONFIG(num_wgs, wg_size, queue_ref)     \
    deep_ep::SyclLaunchConfig __sycl_cfg(num_wgs, wg_size);       \
    auto& __sycl_queue = (queue_ref)

// With local memory
#define SETUP_SYCL_LAUNCH_CONFIG_WITH_LOCAL_MEM(num_wgs, wg_size, local_bytes, queue_ref) \
    deep_ep::SyclLaunchConfig __sycl_cfg(num_wgs, wg_size, local_bytes);                  \
    auto& __sycl_queue = (queue_ref)

// ---------------------------------------------------------------------------
// LAUNCH_SYCL_KERNEL — submit a kernel to the queue
//
// Usage:
//   SETUP_SYCL_LAUNCH_CONFIG(num_eus, 256, queue);
//   LAUNCH_SYCL_KERNEL(__sycl_cfg, __sycl_queue, ([=](sycl::nd_item<1> item) {
//       // kernel body
//   }));
// ---------------------------------------------------------------------------
#define LAUNCH_SYCL_KERNEL(config, queue, kernel_lambda)                    \
    do {                                                                    \
        auto __nd_range = (config).nd_range();                              \
        (queue).submit([&](sycl::handler& cgh) {                           \
            cgh.parallel_for(__nd_range, kernel_lambda);                    \
        });                                                                 \
    } while (0)

// With local memory accessor
#define LAUNCH_SYCL_KERNEL_WITH_LOCAL_MEM(config, queue, local_type, local_name, kernel_lambda) \
    do {                                                                                        \
        auto __nd_range = (config).nd_range();                                                  \
        (queue).submit([&](sycl::handler& cgh) {                                               \
            sycl::local_accessor<local_type, 1> local_name(                                     \
                sycl::range<1>((config).local_mem_bytes / sizeof(local_type)), cgh);            \
            cgh.parallel_for(__nd_range, kernel_lambda);                                        \
        });                                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Dispatch macros — switch over num_ranks, hidden, dtype, etc.
// These are identical in structure to the CUDA versions but use P2P naming.
// ---------------------------------------------------------------------------

#define SWITCH_RANKS(case_macro)                           \
    switch (num_ranks) {                                   \
        case 2:                                            \
            case_macro(2);                                 \
        case 4:                                            \
            case_macro(4);                                 \
        case 8:                                            \
            case_macro(8);                                 \
        default:                                           \
            EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                      \
    while (false)

#define SWITCH_RDMA_RANKS(case_macro)                           \
    switch (num_ranks / NUM_MAX_P2P_PEERS) {                    \
        case 2:                                                 \
            case_macro(2);                                      \
        case 3:                                                 \
            case_macro(3);                                      \
        case 4:                                                 \
            case_macro(4);                                      \
        case 6:                                                 \
            case_macro(6);                                      \
        case 8:                                                 \
            case_macro(8);                                      \
        case 12:                                                \
            case_macro(12);                                     \
        case 16:                                                \
            case_macro(16);                                     \
        case 18:                                                \
            case_macro(18);                                     \
        case 20:                                                \
            case_macro(20);                                     \
        default:                                                \
            EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    }                                                           \
    while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)         \
    switch (num_ranks) {                                   \
        case 2:                                            \
            case_macro(dtype, 2);                          \
        case 4:                                            \
            case_macro(dtype, 4);                          \
        case 8:                                            \
            case_macro(dtype, 8);                          \
        default:                                           \
            EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                      \
    while (false)

// SYCL uses sycl::half for bfloat16; dtype switch maps to SYCL types
#define SWITCH_TYPES(case_macro)                                \
    switch (type) {                                             \
        case 0: /* bfloat16 */                                  \
            case_macro(sycl::ext::oneapi::bfloat16);            \
        default:                                                \
            EP_HOST_ASSERT(false and "Unsupported data type");  \
    }                                                           \
    while (false)

#define SWITCH_HIDDEN(case_macro)                           \
    switch (hidden) {                                       \
        case 2048:                                          \
            case_macro(2048);                               \
        case 2560:                                          \
            case_macro(2560);                               \
        case 3072:                                          \
            case_macro(3072);                               \
        case 4096:                                          \
            case_macro(4096);                               \
        case 5120:                                          \
            case_macro(5120);                               \
        case 6144:                                          \
            case_macro(6144);                               \
        case 7168:                                          \
            case_macro(7168);                               \
        case 8192:                                          \
            case_macro(8192);                               \
        default:                                            \
            EP_HOST_ASSERT(false and "Unsupported hidden"); \
    }                                                       \
    while (false)

}  // namespace deep_ep
