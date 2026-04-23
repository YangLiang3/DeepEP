#pragma once

// SYCL port of kernels/buffer.cuh
//
// Device-side buffer helpers for laying out data within a contiguous allocation.
// Buffer<T>     — simple linear buffer carved from a global pointer
// AsymBuffer<T> — per-rank asymmetric buffer (separate pointer per rank)
// SymBuffer<T>  — symmetric send/recv buffer pair

#include <cstdint>
#include "sycl_backend/configs.hpp"
#include "sycl_backend/exception.hpp"

namespace deep_ep {

// ---------------------------------------------------------------------------
// DeviceBuffer — simple typed view over a device allocation
// (named DeviceBuffer to avoid conflict with the host-side Buffer class)
// ---------------------------------------------------------------------------
template <typename dtype_t>
struct DeviceBuffer {
private:
    uint8_t* ptr;

public:
    int64_t total_bytes;

    DeviceBuffer() : ptr(nullptr), total_bytes(0) {}

    explicit DeviceBuffer(void*& gbl_ptr, int num_elems, int offset = 0) {
        total_bytes = static_cast<int64_t>(num_elems) * sizeof(dtype_t);
        ptr = static_cast<uint8_t*>(gbl_ptr) + offset * sizeof(dtype_t);
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    DeviceBuffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    dtype_t* buffer() { return reinterpret_cast<dtype_t*>(ptr); }

    dtype_t& operator[](int idx) { return buffer()[idx]; }
};

// ---------------------------------------------------------------------------
// DeviceAsymBuffer — per-rank view into separate or shared allocations
// ---------------------------------------------------------------------------
template <typename dtype_t, int kNumRanks = 1>
struct DeviceAsymBuffer {
private:
    uint8_t* ptrs[kNumRanks];
    int64_t num_bytes;

public:
    int64_t total_bytes;

    // Single-rank case: carve from one global pointer
    DeviceAsymBuffer(void*& gbl_ptr, int num_elems, int num_ranks,
               int eu_id = 0, int num_eus = 1, int offset = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "Single-rank ctor requires kNumRanks == 1");
        num_bytes = static_cast<int64_t>(num_elems) * sizeof(dtype_t);

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_eus;
        ptrs[0] = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * eu_id + num_bytes * offset;
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    // Multi-rank case: array of global pointers (one per rank)
    DeviceAsymBuffer(void** gbl_ptrs, int num_elems, int num_ranks,
               int eu_id = 0, int num_eus = 1, int offset = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "Multi-rank ctor requires kNumRanks > 1");
        num_bytes = static_cast<int64_t>(num_elems) * sizeof(dtype_t);

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_eus;
        for (int i = 0; i < kNumRanks; ++i) {
            ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + per_channel_bytes * eu_id + num_bytes * offset;
            gbl_ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        }
    }

    void advance(int shift) {
        #pragma unroll
        for (int i = 0; i < kNumRanks; ++i)
            ptrs[i] = ptrs[i] + shift * sizeof(dtype_t);
    }

    DeviceAsymBuffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    template <int kNumAlsoRanks>
    DeviceAsymBuffer advance_also(void** gbl_ptrs) {
        for (int i = 0; i < kNumAlsoRanks; ++i)
            gbl_ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        return *this;
    }

    dtype_t* buffer(int idx = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "`buffer` requires kNumRanks == 1");
        return reinterpret_cast<dtype_t*>(ptrs[0] + num_bytes * idx);
    }

    dtype_t* buffer_by(int rank_idx, int idx = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "`buffer_by` requires kNumRanks > 1");
        return reinterpret_cast<dtype_t*>(ptrs[rank_idx] + num_bytes * idx);
    }
};

// ---------------------------------------------------------------------------
// DeviceSymBuffer — symmetric send/recv (or single) buffer
// ---------------------------------------------------------------------------
template <typename dtype_t, bool kDecoupled = true>
struct DeviceSymBuffer {
private:
    uint8_t* send_ptr;
    uint8_t* recv_ptr;  // unused when !kDecoupled
    int64_t num_bytes;

public:
    int64_t total_bytes;

    DeviceSymBuffer(void*& gbl_ptr, int num_elems, int num_ranks,
              int eu_id = 0, int num_eus = 1) {
        num_bytes = static_cast<int64_t>(num_elems) * sizeof(dtype_t);

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_eus * (static_cast<int>(kDecoupled) + 1);
        send_ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * eu_id;
        recv_ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * (eu_id + num_eus);
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    dtype_t* send_buffer(int idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`send_buffer` requires kDecoupled == true");
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);
    }

    dtype_t* recv_buffer(int idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`recv_buffer` requires kDecoupled == true");
        return reinterpret_cast<dtype_t*>(recv_ptr + num_bytes * idx);
    }

    dtype_t* buffer(int idx = 0) {
        EP_STATIC_ASSERT(not kDecoupled, "`buffer` requires kDecoupled == false");
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);
    }
};

}  // namespace deep_ep
