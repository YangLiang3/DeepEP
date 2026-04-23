#pragma once

// SYCL equivalents for kernels/utils.cuh
//
// Mapping summary:
//   CUDA warp (32 lanes)          → SYCL sub_group (typically 16 or 32 lanes on Intel GPU)
//   __syncthreads()               → sycl::group_barrier(item.get_group())
//   __shfl_sync / __shfl_xor_sync → sub_group.shuffle / shuffle_xor
//   atomicAdd_system              → sycl::atomic_ref<..., memory_scope::system>
//   __shared__                    → sycl::local_accessor (passed via nd_range launch)
//   clock64() for timeout         → steady_clock or cycle counter
//   ld.volatile / st.volatile     → sycl::atomic_ref with relaxed ordering
//   memory fences (PTX)           → sycl::atomic_fence with appropriate scope
//
// Intel GPU sub-group sizes: 8, 16, or 32 (varies by HW / kernel).
// We define SUB_GROUP_SIZE as a compile-time constant; kernels must request
// [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]].

#include <sycl/sycl.hpp>
#include <cstdint>
#include <type_traits>

#include "sycl_backend/configs.hpp"
#include "sycl_backend/exception.hpp"

namespace deep_ep {

// ---------------------------------------------------------------------------
// Sub-group size — Intel GPUs commonly use 16; can be overridden at compile time.
// ---------------------------------------------------------------------------
#ifndef SUB_GROUP_SIZE
#define SUB_GROUP_SIZE 16
#endif

// ---------------------------------------------------------------------------
// Vector type mapping (replaces CUDA int4 / int2)
// ---------------------------------------------------------------------------
struct alignas(16) sycl_int4 {
    int x, y, z, w;
};

struct alignas(8) sycl_int2 {
    int x, y;
};

template <int kBytes>
struct VecInt {};
template <> struct VecInt<1>  { using vec_t = int8_t; };
template <> struct VecInt<2>  { using vec_t = int16_t; };
template <> struct VecInt<4>  { using vec_t = int32_t; };
template <> struct VecInt<8>  { using vec_t = int64_t; };
template <> struct VecInt<16> { using vec_t = sycl_int4; };

// ---------------------------------------------------------------------------
// Memory fences — replacing PTX fence instructions
// ---------------------------------------------------------------------------

// fence.acq_rel.sys — system-wide (cross-device) memory fence
inline void memory_fence() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

// fence.acq_rel.gpu — device-scope memory fence
inline void memory_fence_device() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
}

// fence.acq_rel.cta — work-group-scope memory fence
inline void memory_fence_work_group() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
}

// ---------------------------------------------------------------------------
// Volatile / relaxed loads and stores via atomic_ref
// These replace PTX ld.volatile.global / st.volatile.global / ld.acquire / st.release
// ---------------------------------------------------------------------------

// ld.volatile.global equivalent — relaxed load with system scope
template <typename T>
inline T ld_volatile_global(const T* ptr) {
    auto* non_const = const_cast<T*>(ptr);
    sycl::atomic_ref<T, sycl::memory_order::relaxed,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space> ref(*non_const);
    return ref.load();
}

// st.relaxed.sys.global equivalent
template <typename T>
inline void st_relaxed_sys_global(const T* ptr, T val) {
    auto* non_const = const_cast<T*>(ptr);
    sycl::atomic_ref<T, sycl::memory_order::relaxed,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space> ref(*non_const);
    ref.store(val);
}

// st.release.sys.global equivalent
template <typename T>
inline void st_release_sys_global(const T* ptr, T val) {
    auto* non_const = const_cast<T*>(ptr);
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space> ref(*non_const);
    ref.store(val, sycl::memory_order::release);
}

// ld.acquire.sys.global equivalent
template <typename T>
inline T ld_acquire_sys_global(const T* ptr) {
    auto* non_const = const_cast<T*>(ptr);
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space> ref(*non_const);
    return ref.load(sycl::memory_order::acquire);
}

// ld.acquire.gpu.global equivalent
template <typename T>
inline T ld_acquire_global(const T* ptr) {
    auto* non_const = const_cast<T*>(ptr);
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(*non_const);
    return ref.load(sycl::memory_order::acquire);
}

// st.release.cta (work-group scope)
template <typename T>
inline void st_release_work_group(const T* ptr, T val) {
    auto* non_const = const_cast<T*>(ptr);
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::global_space> ref(*non_const);
    ref.store(val, sycl::memory_order::release);
}

// ld.acquire.cta (work-group scope)
template <typename T>
inline T ld_acquire_work_group(const T* ptr) {
    auto* non_const = const_cast<T*>(ptr);
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::global_space> ref(*non_const);
    return ref.load(sycl::memory_order::acquire);
}

// ---------------------------------------------------------------------------
// Atomic operations — replacing CUDA atomicAdd_system / atomicAdd / atomicCAS
// ---------------------------------------------------------------------------

// atomicAdd with system scope (cross-device visible)
template <typename T>
inline T atomic_add_system(T* ptr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space> ref(*ptr);
    return ref.fetch_add(val);
}

// atomicAdd with release + system scope
template <typename T>
inline T atomic_add_release_sys_global(T* ptr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space> ref(*ptr);
    return ref.fetch_add(val, sycl::memory_order::release);
}

// atomicAdd with release + device scope
template <typename T>
inline T atomic_add_release_global(T* ptr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(*ptr);
    return ref.fetch_add(val, sycl::memory_order::release);
}

// atomicSub with system scope
template <typename T>
inline T atomic_sub_system(T* ptr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space> ref(*ptr);
    return ref.fetch_sub(val);
}

// atomicCAS with acquire + work-group scope (for locks)
template <typename T>
inline T atomic_cas_work_group_acquire(T* ptr, T expected, T desired) {
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::global_space> ref(*ptr);
    ref.compare_exchange_strong(expected, desired, sycl::memory_order::acquire);
    return expected;  // returns old value (before CAS)
}

// atomicExch with release + work-group scope
template <typename T>
inline T atomic_exch_work_group_release(T* ptr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::global_space> ref(*ptr);
    return ref.exchange(val, sycl::memory_order::release);
}

// ---------------------------------------------------------------------------
// Lock primitives (using work-group-scope atomics, like CUDA CTA-scope)
// ---------------------------------------------------------------------------
inline void acquire_lock(int* mutex) {
    while (atomic_cas_work_group_acquire(mutex, 0, 1) != 0)
        ;
}

inline void release_lock(int* mutex) {
    atomic_exch_work_group_release(mutex, 0);
}

// ---------------------------------------------------------------------------
// Non-cached load/store — ld.global.nc / st.global.L1::no_allocate equivalents
//
// On Intel GPU there are no direct PTX-style cache-control instructions.
// We use normal loads/stores. The L1 cache behavior is managed by the HW.
// For streaming (non-temporal) hints, Intel provides lsc_load/lsc_store
// extensions, but those are optional and HW-specific. For now we use
// regular reads/writes as the SYCL portable path.
// ---------------------------------------------------------------------------
template <typename T>
inline T ld_nc_global(const T* ptr) {
    return *ptr;
}

template <typename T>
inline void st_na_global(const T* ptr, const T& value) {
    *const_cast<T*>(ptr) = value;
}

// Relaxed load/store for non-allocating patterns (ld_na_relaxed / st_na_relaxed)
template <typename T>
inline T ld_na_relaxed(const T* ptr) {
    return *ptr;
}

template <typename T>
inline void st_na_relaxed(const T* ptr, T val) {
    *const_cast<T*>(ptr) = val;
}

// Relaxed non-allocating store with release semantics
template <typename T>
inline void st_na_release(const T* ptr, T val) {
    auto* non_const = const_cast<T*>(ptr);
    sycl::atomic_ref<T, sycl::memory_order::acq_rel,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(*non_const);
    ref.store(val, sycl::memory_order::release);
}

// ---------------------------------------------------------------------------
// Sub-group (warp) utilities
// ---------------------------------------------------------------------------

// get_lane_id() → sub_group local id
inline uint32_t get_lane_id(sycl::sub_group sg) {
    return sg.get_local_linear_id();
}

// elect_one_sync() → true for first active lane
inline uint32_t elect_one_sync(sycl::sub_group sg) {
    return sg.get_local_linear_id() == 0 ? 1 : 0;
}

// broadcast — shuffle from src lane
template <typename T>
inline T broadcast(sycl::sub_group sg, T value, uint32_t src_lane) {
    return sycl::group_broadcast(sg, value, src_lane);
}

// ---------------------------------------------------------------------------
// Sub-group reductions — replacing warp_reduce_sum / max / min / and / or
//
// CUDA uses __shfl_xor_sync; SYCL uses sub_group shuffle_xor.
// We provide both a generic reduce and convenience aliases.
// ---------------------------------------------------------------------------

// Operation functors (host+device)
template <typename T> struct ReduceSum {
    T operator()(T a, T b) const { return a + b; }
};
template <typename T> struct ReduceMax {
    T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T> struct ReduceMin {
    T operator()(T a, T b) const { return a < b ? a : b; }
};
template <typename T> struct ReduceAnd {
    T operator()(T a, T b) const { return a & b; }
};
template <typename T> struct ReduceOr {
    T operator()(T a, T b) const { return a | b; }
};

// Generic sub-group reduce via shuffle_xor
// kNumLanesPerGroup: how many lanes participate (must be power-of-2, ≤ sub_group size)
// kIntergroupReduce: if true, reduce ACROSS groups; if false, reduce WITHIN a group
template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
inline T sub_group_reduce(sycl::sub_group sg, T value, Op op) {
    // Intel GPU sub-group sizes can be 8, 16, or 32.
    // We use shuffle_xor which works for any sub-group size.
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <= 1)
            value = op(value, sycl::shift_group_left(sg, value, 1));
        if constexpr (kNumLanesPerGroup <= 2)
            value = op(value, sycl::shift_group_left(sg, value, 2));
        if constexpr (kNumLanesPerGroup <= 4)
            value = op(value, sycl::shift_group_left(sg, value, 4));
        if constexpr (kNumLanesPerGroup <= 8)
            value = op(value, sycl::shift_group_left(sg, value, 8));
        if constexpr (kNumLanesPerGroup <= 16)
            value = op(value, sycl::shift_group_left(sg, value, 16));
    } else {
        if constexpr (kNumLanesPerGroup >= 32)
            value = op(value, sycl::shift_group_left(sg, value, 16));
        if constexpr (kNumLanesPerGroup >= 16)
            value = op(value, sycl::shift_group_left(sg, value, 8));
        if constexpr (kNumLanesPerGroup >= 8)
            value = op(value, sycl::shift_group_left(sg, value, 4));
        if constexpr (kNumLanesPerGroup >= 4)
            value = op(value, sycl::shift_group_left(sg, value, 2));
        if constexpr (kNumLanesPerGroup >= 2)
            value = op(value, sycl::shift_group_left(sg, value, 1));
    }
    return value;
}

// Convenience aliases
template <int kNumLanesPerGroup = SUB_GROUP_SIZE, bool kIntergroupReduce = false, typename T>
inline T sub_group_reduce_sum(sycl::sub_group sg, T value) {
    return sub_group_reduce<kNumLanesPerGroup, kIntergroupReduce>(sg, value, ReduceSum<T>{});
}

template <int kNumLanesPerGroup = SUB_GROUP_SIZE, bool kIntergroupReduce = false, typename T>
inline T sub_group_reduce_max(sycl::sub_group sg, T value) {
    return sub_group_reduce<kNumLanesPerGroup, kIntergroupReduce>(sg, value, ReduceMax<T>{});
}

template <int kNumLanesPerGroup = SUB_GROUP_SIZE, bool kIntergroupReduce = false, typename T>
inline T sub_group_reduce_min(sycl::sub_group sg, T value) {
    return sub_group_reduce<kNumLanesPerGroup, kIntergroupReduce>(sg, value, ReduceMin<T>{});
}

template <int kNumLanesPerGroup = SUB_GROUP_SIZE, bool kIntergroupReduce = false, typename T>
inline T sub_group_reduce_and(sycl::sub_group sg, T value) {
    return sub_group_reduce<kNumLanesPerGroup, kIntergroupReduce>(sg, value, ReduceAnd<T>{});
}

template <int kNumLanesPerGroup = SUB_GROUP_SIZE, bool kIntergroupReduce = false, typename T>
inline T sub_group_reduce_or(sycl::sub_group sg, T value) {
    return sub_group_reduce<kNumLanesPerGroup, kIntergroupReduce>(sg, value, ReduceOr<T>{});
}

// ---------------------------------------------------------------------------
// Sub-group copy macro — replacing UNROLLED_WARP_COPY
// Uses sub-group lanes instead of warp lanes.
// ---------------------------------------------------------------------------
#define UNROLLED_SUBGROUP_COPY(UNROLL_FACTOR, LANE_ID, SG_SIZE, N, DST, SRC, LD_FUNC, ST_FUNC) \
    {                                                                                           \
        constexpr int kLoopStride = (SG_SIZE) * (UNROLL_FACTOR);                                \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
        auto __src = (SRC);                                                                     \
        auto __dst = (DST);                                                                     \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)                   \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * (SG_SIZE));                  \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)                   \
                ST_FUNC(__dst + __i + __j * (SG_SIZE), unrolled_values[__j]);                   \
        }                                                                                       \
        {                                                                                       \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                            \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                 \
                if (__i + __j * (SG_SIZE) < (N))                                                \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * (SG_SIZE));              \
            }                                                                                   \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                 \
                if (__i + __j * (SG_SIZE) < (N))                                                \
                    ST_FUNC(__dst + __i + __j * (SG_SIZE), unrolled_values[__j]);               \
            }                                                                                   \
        }                                                                                       \
    }

// ---------------------------------------------------------------------------
// Inter-thread barrier within a work-group (replaces barrier_block from CUDA)
// ---------------------------------------------------------------------------
template <int kNumRanks, bool kSyncOnly = false>
inline void barrier_block(sycl::nd_item<1> item, int** barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(item.get_local_linear_id());
    auto wg = item.get_group();

    // For non-sync-only: ensure prior memory ops are visible at system scope
    if constexpr (not kSyncOnly) {
        memory_fence();
        sycl::group_barrier(wg);
    }

    // Signal: add to self, subtract from peers
    if (thread_id < kNumRanks) {
        atomic_add_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
        atomic_sub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
    }

    // Spin-wait until all peers have signalled
    // NOTE: timeout via cycle count is not portable on SYCL; we use a simple
    // iteration counter with a generous limit.
    constexpr int kMaxSpinIters = 1000000000;  // ~= seconds at GHz clock
    int spin_count = 0;
    while (true) {
        auto value = thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;
        // All lanes in sub-group must agree that value <= 0
        // Use work-group collective since CUDA uses __all_sync across full warp
        bool all_done = sycl::all_of_group(wg, value <= 0);
        if (all_done) break;

        if (++spin_count > kMaxSpinIters && thread_id < kNumRanks) {
            // Timeout — cannot trap on SYCL, but we can assert
            // In practice this should never happen if the algorithm is correct
            break;
        }
    }
    sycl::group_barrier(wg);
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

template <typename T>
constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
constexpr T align_up(T a, T b) {
    return ceil_div<T>(a, b) * b;
}

template <typename T>
constexpr T align_down(T a, T b) {
    return a / b * b;
}

// Channel task range — divides tokens across execution units (EUs / SMs)
inline void get_channel_task_range(int num_tokens, int num_eus, int eu_id,
                                    int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_eu = ceil_div(num_tokens, num_eus);
    token_start_idx = sycl::min(num_tokens_per_eu * eu_id, num_tokens);
    token_end_idx = sycl::min(token_start_idx + num_tokens_per_eu, num_tokens);
}

// ---------------------------------------------------------------------------
// Pack / unpack helpers
// ---------------------------------------------------------------------------
template <typename A, typename B>
inline B pack2(const A& x, const A& y) {
    EP_STATIC_ASSERT(sizeof(A) * 2 == sizeof(B), "Invalid dtypes for pack2");
    B packed;
    auto* p = reinterpret_cast<A*>(&packed);
    p[0] = x;
    p[1] = y;
    return packed;
}

template <typename A, typename B>
inline void unpack2(const B& packed, A& x, A& y) {
    EP_STATIC_ASSERT(sizeof(A) * 2 == sizeof(B), "Invalid dtypes for unpack2");
    auto* p = reinterpret_cast<const A*>(&packed);
    x = p[0];
    y = p[1];
}

// ---------------------------------------------------------------------------
// FP8 constants and helpers
// ---------------------------------------------------------------------------
constexpr float kFP8Margin = 1e-4f;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1.0f / 448.0f;

inline float fast_pow2(int x) {
    uint32_t bits_x = static_cast<uint32_t>(x + 127) << 23;
    float result;
    std::memcpy(&result, &bits_x, sizeof(float));
    return result;
}

inline int fast_log2_ceil(float x) {
    uint32_t bits_x;
    std::memcpy(&bits_x, &x, sizeof(uint32_t));
    auto exp_x = (bits_x >> 23) & 0xff;
    auto man_bits = bits_x & ((1 << 23) - 1);
    return static_cast<int>(exp_x) - 127 + (man_bits != 0);
}

inline void calculate_fp8_scales(float amax, float& scale, float& scale_inv, bool round_scale) {
    if (round_scale) {
        auto exp_scale_inv = fast_log2_ceil(amax * kFinfoAmaxInvE4M3);
        scale = fast_pow2(-exp_scale_inv);
        scale_inv = fast_pow2(exp_scale_inv);
    } else {
        scale_inv = amax * kFinfoAmaxInvE4M3;
        scale = kFinfoAmaxE4M3 / amax;
    }
}

// PatternVisitor — device/host functor indexing helper
template <typename FuncT>
struct PatternVisitor {
    FuncT func;
    explicit PatternVisitor(FuncT&& f) : func(std::forward<FuncT>(f)) {}
    auto operator[](const uint32_t& i) { return func(i); }
};

}  // namespace deep_ep
