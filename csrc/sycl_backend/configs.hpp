#pragma once

// SYCL-compatible version of kernels/configs.cuh
// Constants are shared between CUDA and SYCL backends.

#define NUM_MAX_NVL_PEERS 8
#define NUM_MAX_RDMA_PEERS 20
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_WAIT_NANOSECONDS 500

#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s
#else
#define NUM_CPU_TIMEOUT_SECS 10
#define NUM_TIMEOUT_CYCLES 20000000000ull  // 20G cycles ~= 10s
#endif

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

#include <cstdint>

namespace deep_ep {

// topk_idx type: matches the CUDA backend
#if defined(TOPK_IDX_BITS) && TOPK_IDX_BITS == 64
using topk_idx_t = int64_t;
#else
using topk_idx_t = int32_t;
#endif

}  // namespace deep_ep
