#pragma once

// SYCL + iSHMEM IBGDA device-side abstraction layer for DeepEP.
//
// This file provides wrapper functions that map NVSHMEM IBGDA device-side
// symbols (used by the CUDA kernels) to iSHMEM public APIs.
//
// Design principles:
//   1. Use iSHMEM public APIs (ishmem_*, ishmemx_*) rather than internal
//      ishmemi_ibgda_* functions where possible.
//   2. Maintain the same call-site semantics as the NVSHMEM originals so that
//      kernel porting is mechanical.
//   3. Sub-group cooperation is handled at the wrapper level: only the leader
//      lane calls into iSHMEM, then the sub-group synchronizes.
//
// NVSHMEM → iSHMEM mapping summary (see doc/TODO.md Phase 5 for full table):
//   nvshmemi_ibgda_put_nbi_warp  → ishmem_put_nbi (leader-only + sub-group barrier)
//   nvshmemi_ibgda_rma_p         → ishmem_int_p (scalar blocking put)
//   nvshmemi_ibgda_amo_nonfetch_add → ishmem_int_atomic_add / local sycl::atomic_ref
//   nvshmemi_ibgda_quiet         → ishmem_quiet (global quiet — waits for ALL PEs)
//   nvshmemi_get_p2p_ptr         → ishmem_ptr (node-local peer pointer)
//   nvshmem_sync_all             → ishmem_sync_all
//   nvshmem_sync(team)           → ishmem_team_sync(team)
//   nvshmemx_barrier_all_block   → ishmemx_barrier_all_work_group(group)

#include <sycl/sycl.hpp>
#include <cstdint>

#include "sycl_backend/configs.hpp"
#include "sycl_backend/exception.hpp"
#include "sycl_backend/sycl_utils.hpp"

// iSHMEM headers — public device-side API
#ifdef SYCL_ISHMEM
#include <ishmem.h>
#include <ishmemx.h>
#endif

namespace deep_ep {

// ---------------------------------------------------------------------------
// Type alias: nvshmem_team_t → ishmem_team_t
// ---------------------------------------------------------------------------
#ifdef SYCL_ISHMEM
using shmem_team_t = ishmem_team_t;
#else
using shmem_team_t = int;
#endif

// ---------------------------------------------------------------------------
// ibgda_put_nbi_subgroup — Non-blocking RDMA PUT (sub-group cooperative)
//
// NVSHMEM original: nvshmemi_ibgda_put_nbi_warp(rptr, lptr, bytes, pe, qp_id, lane_id, msg_idx)
//   - Warp-cooperative: each lane computes its own lkey/rkey chunk, constructs
//     one WQE, and lane 0 reserves slots / submits.
//
// iSHMEM equivalent: ishmem_put_nbi(dest, src, nelems, pe)
//   - Single-thread call. Internally dispatches to direct doorbell
//     (ishmemi_ibgda_device_emit_direct_wqe_skeleton) or staged WQE fallback.
//   - Sub-group cooperation: only the leader calls ishmem_put_nbi, then all
//     lanes synchronize via sub-group barrier.
//
// Parameters kept similar to NVSHMEM for easy kernel porting:
//   req_rptr  — remote destination address (symmetric heap pointer)
//   req_lptr  — local source address (symmetric heap pointer)
//   bytes     — transfer size in bytes
//   dst_pe    — destination PE index
//   qp_id     — QP index (ignored by iSHMEM — uses internal peer_context)
//   lane_id   — sub-group lane ID
//   message_idx — message index (ignored by iSHMEM — no batched doorbell)
// ---------------------------------------------------------------------------
template <bool kAlwaysDoPostSend = false>
inline void ibgda_put_nbi_subgroup(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes,
    int dst_pe, [[maybe_unused]] int qp_id,
    int lane_id, [[maybe_unused]] int message_idx) {
#ifdef SYCL_ISHMEM
    // Only sub-group leader issues the put
    if (lane_id == 0) {
        auto* dst = reinterpret_cast<void*>(req_rptr);
        auto* src = reinterpret_cast<const void*>(req_lptr);
        size_t nelems = bytes;  // byte-granularity: use uint8 overload
        ishmem_uint8_put_nbi(static_cast<uint8_t*>(dst),
                             static_cast<const uint8_t*>(src),
                             nelems, dst_pe);
    }
    // All lanes synchronize (replaces __syncwarp)
    sycl::group_barrier(sycl::ext::oneapi::experimental::this_sub_group());
#else
    (void)req_rptr; (void)req_lptr; (void)bytes;
    (void)dst_pe; (void)lane_id;
#endif
}

// ---------------------------------------------------------------------------
// ibgda_rma_p — Scalar blocking PUT of a single int value
//
// NVSHMEM original: nvshmemi_ibgda_rma_p(rptr, value, pe, qp_id, imm)
//   - Inline RDMA WRITE of 4 bytes with optional IMM data, fire-and-forget.
//
// iSHMEM equivalent: ishmem_int_p(dest, value, pe)
//   - Blocking scalar put. Internally posts RDMA WRITE + polls CQ.
//   - No IMM data support (IMM feature not used by DeepEP in practice).
//
// Note: NVSHMEM version is fire-and-forget (needs separate quiet for completion).
// iSHMEM ishmem_int_p is blocking (data is visible on return). This provides
// stronger guarantees, which is safe for correctness but may have latency
// implications. If needed, the wrapper can be changed to use put_nbi + quiet.
// ---------------------------------------------------------------------------
inline void ibgda_rma_p(
    int* rptr, int value, int pe,
    [[maybe_unused]] int qp_id,
    [[maybe_unused]] uint32_t imm = 0xFFFFFFFF) {
#ifdef SYCL_ISHMEM
    ishmem_int_p(rptr, value, pe);
#else
    (void)rptr; (void)value; (void)pe;
#endif
}

// ---------------------------------------------------------------------------
// ibgda_amo_nonfetch_add — Atomic non-fetching add on remote memory
//
// NVSHMEM original: nvshmemi_ibgda_amo_nonfetch_add(rptr, value, pe, qp_id, is_local_copy)
//   - If is_local_copy: local atomicAdd.
//   - Else: constructs MLX5_OPCODE_ATOMIC_MASKED_FA WQE (32-bit extended AMO).
//
// iSHMEM equivalent: ishmem_int_atomic_add(dest, value, pe)
//   - For remote PEs: internally dispatches to IBGDA device AMO
//     (ishmemi_ibgda_device_amo_nonfetch<int, AMO_ADD>), which uses 64-bit
//     RDMA FADD with CAS-loop for 32-bit subword extraction.
//   - For local copy: we use sycl::atomic_ref directly (faster than going
//     through iSHMEM which would also check for node-local optimization).
// ---------------------------------------------------------------------------
inline void ibgda_amo_nonfetch_add(
    void* rptr, int value, int pe,
    [[maybe_unused]] int qp_id,
    bool is_local_copy = false) {
#ifdef SYCL_ISHMEM
    if (is_local_copy) {
        // Local atomic add — direct sycl::atomic_ref (no RDMA needed)
        auto* iptr = static_cast<int*>(rptr);
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space> ref(*iptr);
        ref.fetch_add(value);
    } else {
        // Remote atomic add via iSHMEM
        ishmem_int_atomic_add(static_cast<int*>(rptr), value, pe);
    }
#else
    (void)rptr; (void)value; (void)pe; (void)is_local_copy;
#endif
}

// ---------------------------------------------------------------------------
// ibgda_quiet — Wait for all outstanding RDMA operations to complete
//
// NVSHMEM original: nvshmemi_ibgda_quiet(dst_pe, qp_id)
//   - Per-PE, per-QP completion: polls single CQ until all submitted WQEs
//     for that (PE, QP) pair are done.
//
// iSHMEM equivalent: ishmem_quiet()
//   - GLOBAL quiet: polls ALL PEs' collapsed CQs + proxy quiet.
//   - This is stronger than NVSHMEM's per-PE quiet but always correct.
//
// Performance note: In NVSHMEM DeepEP kernels, quiet is typically called in a
// loop over all destination PEs anyway (e.g., internode.cu line 138 loops over
// dst_rdma_rank). Replacing per-PE quiets with a single global quiet at the
// end of the loop is semantically equivalent and may be more efficient on
// iSHMEM (single pass over all CQs vs. N passes with 1 CQ each).
//
// For kernel porting, call sites can either:
//   (a) Replace the entire quiet loop with a single ibgda_quiet_all(), or
//   (b) Call ibgda_quiet(pe, qp) at each iteration (maps to global quiet,
//       which is redundant but correct — subsequent calls are fast no-ops
//       when nothing new has been submitted).
// ---------------------------------------------------------------------------
inline void ibgda_quiet(
    [[maybe_unused]] int dst_pe,
    [[maybe_unused]] int qp_id) {
#ifdef SYCL_ISHMEM
    ishmem_quiet();
#else
    (void)dst_pe; (void)qp_id;
#endif
}

// Convenience: global quiet (no per-PE arguments)
inline void ibgda_quiet_all() {
#ifdef SYCL_ISHMEM
    ishmem_quiet();
#endif
}

// ---------------------------------------------------------------------------
// get_p2p_ptr — Resolve intra-node P2P (IPC-mapped) peer pointer
//
// NVSHMEM original: nvshmemi_get_p2p_ptr(ptr, rank, dst_rank)
//   - Returns NVLink-mapped peer address, or 0 if RDMA-only.
//   - Used to decide whether to do direct load/store (P2P) or RDMA put.
//
// iSHMEM equivalent: ishmem_ptr(dest, pe)
//   - Returns non-NULL for node-local PEs with IPC mapping, NULL otherwise.
//   - Caller checks: if non-NULL, use direct memcpy; else use RDMA put.
//
// Note: NVSHMEM version takes (ptr, my_rank, dst_rank) and computes offset
// from heap_base. iSHMEM ishmem_ptr(dest, pe) expects dest to be a symmetric
// heap address on the calling PE, and returns the corresponding address on
// the target PE (if accessible via P2P), or nullptr.
// ---------------------------------------------------------------------------
inline uint64_t get_p2p_ptr(uint64_t ptr, int rank, int dst_rank) {
#ifdef SYCL_ISHMEM
    if (rank == dst_rank)
        return ptr;
    void* peer_ptr = ishmem_ptr(reinterpret_cast<const void*>(ptr), dst_rank);
    return reinterpret_cast<uint64_t>(peer_ptr);  // 0 if not P2P-accessible
#else
    (void)rank; (void)dst_rank;
    return ptr;
#endif
}

// ---------------------------------------------------------------------------
// Synchronization primitives
// ---------------------------------------------------------------------------

// nvshmem_sync_all() → ishmem_sync_all()
inline void shmem_sync_all() {
#ifdef SYCL_ISHMEM
    ishmem_sync_all();
#endif
}

// nvshmem_sync(team) → ishmem_team_sync(team)
inline void shmem_sync(shmem_team_t team) {
#ifdef SYCL_ISHMEM
    ishmem_team_sync(team);
#endif
}

// nvshmem_sync_with_same_gpu_idx — helper from internode.cu
// Dispatches between team sync (low-latency mode) and global sync.
template <bool kLowLatencyMode>
inline void shmem_sync_with_same_gpu_idx(const shmem_team_t& rdma_team) {
#ifdef SYCL_ISHMEM
    if constexpr (kLowLatencyMode) {
        ishmem_team_sync(rdma_team);
    } else {
        ishmem_sync_all();
    }
#else
    (void)rdma_team;
#endif
}

// nvshmemx_barrier_all_block() → ishmemx_barrier_all_work_group(group)
// Must be called by ALL work-items in the work-group.
template <typename Group>
inline void shmemx_barrier_all_work_group(const Group& group) {
#ifdef SYCL_ISHMEM
    ishmemx_barrier_all_work_group(group);
#endif
}

// nvshmem_barrier_all() → ishmem_barrier_all()
// Single-thread device-callable barrier.
inline void shmem_barrier_all() {
#ifdef SYCL_ISHMEM
    ishmem_barrier_all();
#endif
}

// ---------------------------------------------------------------------------
// Quiet / fence (work-group scope)
// ---------------------------------------------------------------------------

// Work-group cooperative quiet — leader polls CQs, all threads barrier.
template <typename Group>
inline void shmemx_quiet_work_group(const Group& group) {
#ifdef SYCL_ISHMEM
    ishmemx_quiet_work_group(group);
#endif
}

// Work-group cooperative fence — ordering without completion guarantee.
template <typename Group>
inline void shmemx_fence_work_group(const Group& group) {
#ifdef SYCL_ISHMEM
    ishmemx_fence_work_group(group);
#endif
}

// Single-thread device-callable quiet.
inline void shmem_quiet() {
#ifdef SYCL_ISHMEM
    ishmem_quiet();
#endif
}

// Single-thread device-callable fence.
inline void shmem_fence() {
#ifdef SYCL_ISHMEM
    ishmem_fence();
#endif
}

// ---------------------------------------------------------------------------
// Team management (host-side only)
// ---------------------------------------------------------------------------

// nvshmem_team_split_strided → ishmem_team_split_strided
// This is a host-only call used during Buffer::sync() for low-latency mode.
inline int shmem_team_split_strided(shmem_team_t parent_team,
                                    int start, int stride, int size,
                                    shmem_team_t* new_team) {
#ifdef SYCL_ISHMEM
    return ishmem_team_split_strided(parent_team, start, stride, size,
                                     nullptr, 0, new_team);
#else
    (void)parent_team; (void)start; (void)stride; (void)size; (void)new_team;
    return -1;
#endif
}

// ---------------------------------------------------------------------------
// PE query helpers (device-callable)
// ---------------------------------------------------------------------------

inline int shmem_my_pe() {
#ifdef SYCL_ISHMEM
    return ishmem_my_pe();
#else
    return 0;
#endif
}

inline int shmem_n_pes() {
#ifdef SYCL_ISHMEM
    return ishmem_n_pes();
#else
    return 1;
#endif
}

}  // namespace deep_ep
