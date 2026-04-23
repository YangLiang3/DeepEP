# DeepEP iSHMEM + SYCL Backend Integration TODO

## Goal

Replace CUDA + NVSHMEM with **SYCL + iSHMEM** to deliver DeepEP MoE expert-parallel
communication (dispatch / combine, both high-throughput and low-latency) on Intel GPUs.

## Architecture Overview

```
Python API  (deep_ep/buffer.py)  — unchanged public interface
    │
    ▼
C++ runtime (csrc/deep_ep.cpp)   — host-side orchestration
    │
    ├── CUDA path (existing)     — internode.cu, internode_ll.cu, intranode.cu
    │                               uses NVSHMEM IBGDA device APIs
    │
    └── SYCL path (new)          — sycl/internode.cpp, sycl/internode_ll.cpp, sycl/intranode.cpp
                                    uses iSHMEM IBGDA device APIs
```

Key facts that drive the design:
- DeepEP CUDA kernels call **NVSHMEM IBGDA device-side symbols** directly from GPU
  (`nvshmemi_ibgda_put_nbi_warp`, `nvshmemi_ibgda_quiet`, `nvshmemi_ibgda_amo_nonfetch_add`,
   `nvshmemi_get_p2p_ptr`, `nvshmem_sync_all`, `nvshmemx_barrier_all_block`).
- iSHMEM provides equivalent IBGDA device-side symbols for SYCL
  (`ishmemi_ibgda_device_emit_direct_wqe_skeleton`, `ishmemi_ibgda_device_ring_doorbell`,
   `ishmem_put`, `ishmemx_put_work_group`, `ishmem_barrier_all`).
- CUDA kernels (`__global__`, warp intrinsics, `__shared__`) must be rewritten as
  SYCL kernels (`sycl::nd_range`, sub-group intrinsics, local memory).
- Host runtime (`init/finalize/alloc/free/barrier/team`) is a 1:1 mapping from
  NVSHMEM → iSHMEM.

---

## Phase 0: Project Setup ✅
- [x] Define backend terminology (`cuda_nvshmem`, `sycl_ishmem`).
- [x] Document target hardware/software matrix.
- [x] Add `doc/` folder with design and TODO.

## Phase 1: Host Runtime Abstraction ✅
- [x] Create `internode_runtime_adapter.hpp` with backend-neutral runtime API.
- [x] Refactor `deep_ep.cpp` to use adapter instead of direct `internode::` calls.
- [x] Add capability flags in Python (`get_backend_capabilities`).

## Phase 2: Build System — SYCL + iSHMEM Toolchain ✅
- [x] Add setup.py path for Intel DPC++ (icpx) compiler when `DEEPEP_INTERNODE_BACKEND=ishmem`.
- [x] Detect `ISHMEM_DIR` environment variable and validate iSHMEM installation.
- [x] Link against `libishmem` + Level Zero runtime (`-lze_loader`).
- [x] Add `SYCL_ISHMEM` compile definition (analogous to existing `DISABLE_NVSHMEM`).
- [x] Create SYCL backend stub (`csrc/sycl_backend/deep_ep_sycl.cpp`) with full pybind11 API surface.
- [x] Create SYCL-compatible config/exception headers (`csrc/sycl_backend/configs.hpp`, `exception.hpp`).
- [x] Verify clean compile: `python setup.py build` succeeds with `DEEPEP_INTERNODE_BACKEND=ishmem`.

Note: Directory named `csrc/sycl_backend/` (not `csrc/sycl/`) to avoid shadowing
system `<sycl/...>` headers when `-I csrc/` is on the include path.

## Phase 3: iSHMEM Host Runtime Integration ✅
- [x] Implement `sycl_context.hpp` — SYCL device/queue management singleton
  - Lazy singleton per device ordinal; enumerates Intel GPUs.
  - In-order queues (default + comm); exposes L0 native handles.
  - `DEEPEP_DEVICE_ORDINAL` env var for device selection.
- [x] Implement `l0_ipc_memory.hpp` — Level Zero IPC memory allocator
  - `L0IpcMemHandle` struct (ze_ipc_mem_handle_t + size).
  - `L0MemoryAllocator` — malloc_device / free / IPC get/open/close.
  - `L0HostAllocator` — host-pinned USM allocations.
- [x] Implement `ishmem_runtime.hpp` — iSHMEM host runtime wrapper
  - `init()` / `init_attr()` / `finalize()` for lifecycle.
  - `alloc()` / `malloc()` / `free()` for symmetric heap.
  - `barrier_all()` / `sync_all()` for synchronization.
  - `ptr()` for peer pointer resolution.
  - `team_split_strided()` for sub-team creation.
- [x] Wire real Buffer implementation in `deep_ep_sycl.cpp`:
  - Constructor: allocate NVL buffers via L0 IPC, workspace, host-pinned counters.
  - `sync()`: open remote IPC handles, init iSHMEM, alloc RDMA buffer from symmetric heap.
  - `destroy()`: barrier, close IPC, free iSHMEM/L0 resources, finalize.
- [x] Verify clean compile with `DEEPEP_INTERNODE_BACKEND=ishmem`.

## Phase 4: SYCL Kernel Infrastructure ✅
- [x] Create `csrc/sycl_backend/sycl_utils.hpp` — SYCL port of `kernels/utils.cuh`:
  - Memory fences: `memory_fence()` / `memory_fence_device()` / `memory_fence_work_group()` via `sycl::atomic_fence`.
  - Volatile/relaxed loads/stores: `ld_volatile_global` / `st_relaxed_sys_global` / `ld_acquire_sys_global` etc. via `sycl::atomic_ref`.
  - Atomic operations: `atomic_add_system` / `atomic_sub_system` / CAS / exchange with correct scopes.
  - Lock primitives: `acquire_lock` / `release_lock` using work-group-scope CAS.
  - Sub-group reductions: `sub_group_reduce_sum/max/min/and/or` via `sycl::shift_group_left`.
  - Sub-group utilities: `get_lane_id` / `elect_one_sync` / `broadcast`.
  - Vector types: `sycl_int4` / `sycl_int2` / `VecInt<>` trait.
  - Copy macro: `UNROLLED_SUBGROUP_COPY` (replaces `UNROLLED_WARP_COPY`).
  - Barrier: `barrier_block<kNumRanks>` using system-scope atomics + `sycl::group_barrier`.
  - Arithmetic: `ceil_div` / `align_up` / `align_down` / `get_channel_task_range`.
  - FP8 helpers: `fast_pow2` / `fast_log2_ceil` / `calculate_fp8_scales`.
  - Key design: `sycl::atomic_ref` default order must be `relaxed`/`acq_rel`/`seq_cst`; actual acquire/release specified at call sites.
- [x] Create `csrc/sycl_backend/sycl_buffer.hpp` — SYCL port of `kernels/buffer.cuh`:
  - `DeviceBuffer<T>` — linear buffer carved from global pointer.
  - `DeviceAsymBuffer<T, kNumRanks>` — per-rank asymmetric buffer.
  - `DeviceSymBuffer<T, kDecoupled>` — symmetric send/recv buffer pair.
  - Named `Device*` to avoid conflict with host-side `Buffer` class.
- [x] Create `csrc/sycl_backend/sycl_launch.hpp` — SYCL port of `kernels/launch.cuh`:
  - `SyclLaunchConfig` struct (num_work_groups, work_group_size, local_mem_bytes).
  - `SETUP_SYCL_LAUNCH_CONFIG` / `LAUNCH_SYCL_KERNEL` / `LAUNCH_SYCL_KERNEL_WITH_LOCAL_MEM` macros.
  - Dispatch macros: `SWITCH_RANKS` / `SWITCH_RDMA_RANKS` / `SWITCH_TYPES` / `SWITCH_HIDDEN` (uses `NUM_MAX_P2P_PEERS`).
- [x] Add `EP_DEVICE_ASSERT` macro to `exception.hpp` using `assert()` for SYCL kernels.
- [x] Verify clean compile with `icpx -fsycl`: all headers included in `deep_ep_sycl.cpp`.

## Phase 5: SYCL iSHMEM IBGDA Device-Side Abstraction ✅
- [x] Create `csrc/sycl_backend/ibgda_device.hpp` — SYCL IBGDA device API wrapper:
  - `ibgda_put_nbi_subgroup()` — maps to `ishmem_put_nbi()` (public API, internally dispatches to
    `ishmemi_ibgda_device_post_put_nbi()` → `ishmemi_ibgda_device_emit_direct_wqe_skeleton(PUT)`
    or staged WQE fallback). Sub-group cooperative put handled via leader-only call.
  - `ibgda_quiet()` — maps to `ishmem_quiet()` (public API, internally calls
    `ishmemi_ibgda_device_quiet()` which polls all PEs' collapsed CQs, then proxy quiet).
    Per-PE quiet (`nvshmemi_ibgda_quiet(pe, qp)`) has no direct iSHMEM equivalent;
    `ishmem_quiet()` waits for ALL outstanding operations across ALL PEs.
  - `ibgda_amo_nonfetch_add()` — maps to `ishmem_int_atomic_add(dest, val, pe)` (public API,
    internally dispatches to `ishmemi_ibgda_device_amo_nonfetch<int, AMO_ADD>()` →
    `ishmemi_ibgda_device_rdma_atomic64()` with `MLX5_OPCODE_ATOMIC_FA` for 64-bit types,
    or CAS loop on 8-byte aligned container for 32-bit `int`). For local-copy case,
    use `sycl::atomic_ref::fetch_add()` directly.
  - `ibgda_rma_p()` — maps to `ishmem_int_p(dest, val, pe)` (public API, internally dispatches
    to `ishmemi_ibgda_device_post_put()` which does blocking RDMA WRITE + CQ poll for
    completion, or staged WQE fallback). This is a single-element blocking put.
  - `get_p2p_ptr()` — maps to `ishmem_ptr(dest, pe)` for intra-node peer pointer resolution.
    Returns non-NULL for node-local PEs (L0 IPC mapped), NULL for remote PEs.
  - `barrier_all_block()` — maps to `ishmemx_barrier_all_work_group(group)` (work-group scope).
  - `sync_all()` — maps to `ishmem_sync_all()` (device-callable).
  - `sync_team()` — maps to `ishmem_team_sync(team)` (device-callable).
  - `quiet_work_group()` — maps to `ishmemx_quiet_work_group(group)` (work-group scope;
    leader calls `ishmemi_ibgda_device_quiet()`, falls back to proxy if needed).
  - `fence_work_group()` — maps to `ishmemx_fence_work_group(group)` (work-group scope).

### NVSHMEM → iSHMEM Device API Mapping Table

Key design difference: NVSHMEM IBGDA in DeepEP uses **internal** symbols
(`nvshmemi_ibgda_*`) that directly construct mlx5 WQEs and manage QP state.
iSHMEM exposes **public** APIs (`ishmem_*` / `ishmemx_*`) that internally
dispatch to IBGDA direct doorbell path when available, with automatic fallback
to proxy/staged WQE path. The wrapper should use iSHMEM public APIs rather
than calling internal `ishmemi_ibgda_*` functions directly.

#### RMA Operations

| NVSHMEM (CUDA) | iSHMEM (SYCL) | Notes |
|---|---|---|
| `nvshmemi_ibgda_put_nbi_warp(rptr, lptr, bytes, pe, qp_id, lane_id, msg_idx)` | `ishmem_put_nbi(dest, src, nelems, pe)` | NVSHMEM: warp-cooperative, multi-chunk lkey/rkey management, per-lane WQE construction. iSHMEM: single-thread call, internally handles WQE emission. Sub-group cooperation must be done at wrapper level (leader-only call + barrier). |
| `nvshmemi_ibgda_rma_p(rptr, value, pe, qp_id, imm)` | `ishmem_int_p(dest, value, pe)` | NVSHMEM: inline RDMA WRITE with optional IMM data, fire-and-forget. iSHMEM: blocking scalar put (CQ-polled completion). No IMM data support. |
| (N/A — NVSHMEM uses explicit lkey/rkey via `ibgda_get_lkey_and_rkey`) | (handled internally by iSHMEM `emit_direct_wqe_skeleton` via `peer_ctx->lkey_be/rkey_be`) | iSHMEM uses pre-provisioned per-peer lkey/rkey; no per-operation key lookup needed. |

#### Atomic Operations

| NVSHMEM (CUDA) | iSHMEM (SYCL) | Notes |
|---|---|---|
| `nvshmemi_ibgda_amo_nonfetch_add(rptr, value, pe, qp_id, is_local_copy)` | `ishmem_int_atomic_add(dest, value, pe)` or `ishmem_atomic_add<T>(dest, value, pe)` | NVSHMEM: uses `MLX5_OPCODE_ATOMIC_MASKED_FA` (32-bit extended AMO) + explicit ibuf management. iSHMEM: uses `ishmemi_ibgda_device_amo_nonfetch<T, AMO_ADD>` → 64-bit `MLX5_OPCODE_ATOMIC_FA` with CAS-loop for 32-bit types. For `is_local_copy==true`, both use local atomicAdd. |

#### Completion / Ordering

| NVSHMEM (CUDA) | iSHMEM (SYCL) | Notes |
|---|---|---|
| `nvshmemi_ibgda_quiet(pe, qp_id)` | `ishmem_quiet()` | **Critical difference**: NVSHMEM quiet is per-PE per-QP (polls single CQ). iSHMEM quiet is **global** (polls ALL PEs' CQs). Wrapper may need to use `ishmem_quiet()` at synchronization points where original code quiets all PEs in a loop anyway. |
| (N/A) | `ishmemx_quiet_work_group(group)` | Work-group cooperative quiet: leader polls CQs, all threads barrier after. Use this in work-group-scoped kernels. |
| (N/A) | `ishmem_fence()` / `ishmemx_fence_work_group(group)` | Ordering without completion guarantee. Useful between puts to same PE. |

#### Synchronization

| NVSHMEM (CUDA) | iSHMEM (SYCL) | Notes |
|---|---|---|
| `nvshmem_sync_all()` | `ishmem_sync_all()` | 1:1 mapping. Device-callable. |
| `nvshmem_sync(team)` | `ishmem_team_sync(team)` | 1:1 mapping. Device-callable. |
| `nvshmemx_barrier_all_block()` | `ishmemx_barrier_all_work_group(group)` | NVSHMEM: implicit block scope. iSHMEM: requires explicit `group` parameter (SYCL work-group or sub-group). |
| `nvshmem_team_split_strided(...)` | `ishmem_team_split_strided(...)` | 1:1 mapping. Host-only. |

#### Peer Memory Access

| NVSHMEM (CUDA) | iSHMEM (SYCL) | Notes |
|---|---|---|
| `nvshmemi_get_p2p_ptr(ptr, rank, dst_rank)` | `ishmem_ptr(dest, pe)` | NVSHMEM: returns mapped NVLink P2P address or 0 if RDMA-only. iSHMEM: returns non-NULL for node-local PEs with IPC mapping, NULL otherwise. Caller uses returned pointer for direct load/store (intra-node) or falls back to RDMA put (inter-node). |

#### Internal Infrastructure (NOT directly mapped — handled by iSHMEM internally)

| NVSHMEM Internal | iSHMEM Internal Equivalent | Notes |
|---|---|---|
| `ibgda_get_state()` → `nvshmemi_ibgda_device_state_d` | `ishmemi_ibgda_device_get_context()` → `ishmemi_ibgda_device_context_t*` | Device-side global state. NVSHMEM: flat struct in `__device__` memory. iSHMEM: pointer via `global_info->ibgda_device_ctx`. |
| `ibgda_get_rc(pe, qp_id)` → `nvshmemi_ibgda_device_qp_t*` | `ishmemi_ibgda_device_peer_context_qp(ctx, pe, qp_idx)` → `ishmemi_ibgda_peer_context_t*` | Per-PE QP handle. NVSHMEM: flat RC array indexed by PE × num_rc. iSHMEM: `peers[]` array indexed by PE × num_qps_per_pe. |
| `ibgda_reserve_wqe_slots(qp, n)` | Atomic `fetch_add` on `peer_ctx->nic_wq_cnt_addr` | SQ slot reservation. Both use atomic increment. |
| `ibgda_get_wqe_ptr(qp, idx)` | `peer_ctx->nic_wq_base_addr + (slot << 6)` | WQE buffer address calculation. |
| `ibgda_write_rdma_write_wqe(...)` | Inline in `ishmemi_ibgda_device_emit_direct_wqe_skeleton()` | WQE construction. NVSHMEM: separate function. iSHMEM: monolithic inline function. |
| `ibgda_submit_requests()` → dbr + doorbell | `ishmemi_ibgda_device_ring_doorbell()` | Doorbell ringing. NVSHMEM: lock + dbr update + BAR0 write. iSHMEM: dbr write + UAR MMIO write, optional batching via `db_batch_size`. |
| `ibgda_poll_cq(cq, idx)` | `ishmemi_ibgda_device_quiet()` polls via `ishmemi_ibgda_uc_load16()` | CQ polling. NVSHMEM: per-QP CQ poll. iSHMEM: per-PE collapsed CQ poll with UC load (bypasses GPU cache). |
| `HtoBE64/32/16()` (PTX `prmt` instruction) | `ishmemi_ibgda_htobe64/32()` (pure C++ bit manipulation) | Byte swap. NVSHMEM: PTX intrinsics. iSHMEM: portable C++. |

Acceptance: IBGDA wrapper compiles and iSHMEM device context is accessible from SYCL kernel.
Wrapper uses iSHMEM public APIs where possible; internal APIs documented for reference only.

## Phase 6: Internode High-Throughput Kernels (SYCL)
Port `csrc/kernels/internode.cu` (2377 lines) to `csrc/sycl/internode.cpp`:
- [ ] Port `notify_dispatch` kernel — metadata exchange via IBGDA put + barrier.
- [ ] Port `dispatch` kernel — token packing + RDMA put via IBGDA.
- [ ] Port `dispatch_combine_reply` kernel — combine data return path.
- [ ] Port `notify_combine` kernel — combine metadata exchange.
- [ ] Port `combine` kernel — combine data read-back.
- [ ] Port all template specializations (`kLowLatencyMode`, `kNumRDMARanks`).

Key porting patterns:
- `__global__ void kernel(...)` → `q.submit([&](sycl::handler& h) { h.parallel_for(...) })`
- `blockIdx.x / threadIdx.x` → `nd_item.get_group(0) / nd_item.get_local_id(0)`
- `warp_id = threadIdx.x / 32` → `sub_group sg = nd_item.get_sub_group(); sg.get_group_id()`
- `__shfl_sync` → `sycl::select_from_group(sg, val, id)`
- `__syncthreads()` → `sycl::group_barrier(nd_item.get_group())`
- `SymBuffer<T>` NVSHMEM symmetric buffer → iSHMEM symmetric heap pointer math

Acceptance: `internode_dispatch` and `internode_combine` produce correct results on 2+ Intel GPU nodes.

## Phase 7: Low-Latency Kernels (SYCL)
Port `csrc/kernels/internode_ll.cu` (1292 lines) to `csrc/sycl/internode_ll.cpp`:
- [ ] Port `clean_low_latency_buffer` — barrier + memset.
- [ ] Port `dispatch` kernel — FP8 cast + per-expert IBGDA put + atomic signaling.
- [ ] Port `combine` kernel — data readback + weight application.
- [ ] Port barrier implementation (IBGDA quiet + atomic counter sync).
- [ ] Port timeout/mask handling for unhealthy peers.

Acceptance: `low_latency_dispatch` and `low_latency_combine` produce correct results.

## Phase 8: Intranode Kernels (SYCL)
Port `csrc/kernels/intranode.cu` (1103 lines) to `csrc/sycl/intranode.cpp`:
- [ ] Port NVLink barrier → Intel GPU P2P barrier (L0 IPC or shared memory).
- [ ] Port `notify_dispatch` / `dispatch` / `combine` intranode data movement.
- [ ] Replace `cudaIpcMemHandle` with Level Zero IPC memory handles.

Acceptance: Intranode dispatch/combine works with 8 Intel GPUs on single node.

## Phase 9: Layout and Utility Kernels (SYCL)
- [ ] Port `csrc/kernels/layout.cu` (153 lines) to `csrc/sycl/layout.cpp`.
- [ ] Port remaining utility kernels (prefix sum, counting, etc.).

Acceptance: `get_dispatch_layout` produces correct results.

## Phase 10: Python Integration and Feature Parity
- [ ] Wire SYCL kernel launches through `deep_ep.cpp` → pybind11 bindings.
- [ ] Support `torch.xpu` tensors as input/output (via Intel Extension for PyTorch).
- [ ] Map `at::cuda::CUDAStream` → `sycl::queue` for asynchronous execution.
- [ ] Ensure `Buffer.dispatch()`, `Buffer.combine()`, `Buffer.low_latency_dispatch()`,
      `Buffer.low_latency_combine()` work end-to-end.
- [ ] FP8 support: map `__nv_fp8_storage_t` / `cuda_fp8.h` types to
      SYCL FP8 types or manual bit-manipulation equivalents.
- [ ] Validate `EventHandle` equivalent with SYCL events.

Acceptance: Python-level API produces identical numerical results to CUDA path.

## Phase 11: Testing
- [ ] Port DeepEP test suite to run on Intel GPU.
- [ ] Add 2-node iSHMEM internode dispatch/combine test.
- [ ] Add low-latency mode test.
- [ ] Add FP8 + BF16 precision parity tests.
- [ ] Add timeout/mask behavior test.
- [ ] Add performance regression benchmarks.

## Phase 12: CI and Documentation
- [ ] Add CI job for Intel GPU builds.
- [ ] Update README with Intel GPU build/run instructions.
- [ ] Document NVSHMEM → iSHMEM API mapping for contributors.
- [ ] Publish benchmark comparison (NVIDIA vs Intel GPU) if applicable.

---

## Files to Create (SYCL backend)

```
csrc/sycl_backend/
├── configs.hpp           # SYCL-compatible constants (from configs.cuh)        ✅
├── exception.hpp         # Host-side assertion/error macros                    ✅
├── deep_ep_sycl.cpp      # pybind11 module entry point (stub)                 ✅
├── utils.hpp             # SYCL sub-group / atomic / barrier utilities
├── buffer.hpp            # Symmetric buffer helpers for iSHMEM heap
├── ibgda_device.hpp      # iSHMEM IBGDA device-side API wrapper
├── launch.hpp            # SYCL kernel launch helpers
├── internode.cpp         # Internode dispatch/combine kernels
├── internode_ll.cpp      # Low-latency dispatch/combine kernels
├── intranode.cpp         # Intranode dispatch/combine kernels
├── layout.cpp            # Layout computation kernels
└── runtime.cpp           # iSHMEM host runtime (init/alloc/barrier/finalize)
```

## Files to Modify

| File                                    | Changes                                                |
|-----------------------------------------|--------------------------------------------------------|
| `setup.py`                              | SYCL compiler flags, iSHMEM linking, source list       |
| `csrc/deep_ep.cpp`                      | SYCL queue management, dispatch to SYCL kernels        |
| `csrc/deep_ep.hpp`                      | Add SYCL types, Intel GPU memory handles               |
| `csrc/internode_runtime_adapter.hpp`    | Add iSHMEM runtime implementation                      |
| `csrc/config.hpp`                       | Backend-conditional includes                           |
| `deep_ep/buffer.py`                     | Enable iSHMEM capabilities, `torch.xpu` support       |
| `deep_ep/__init__.py`                   | Export backend constants                               |
