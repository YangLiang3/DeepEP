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

## Phase 3: iSHMEM Host Runtime Integration
- [ ] Implement iSHMEM path in `internode_runtime_adapter.hpp`:
  - `ishmem_init()` / `ishmemx_init_attr()` for initialization.
  - `ishmem_align()` / `ishmem_malloc()` for symmetric heap allocation.
  - `ishmem_free()` for deallocation.
  - `ishmem_barrier_all()` for global barrier.
  - `ishmem_finalize()` for teardown.
  - `ishmem_team_split_strided()` for RDMA sub-team creation.
- [ ] Implement SYCL `sycl::queue` management and device selection in `deep_ep.cpp`.
- [ ] Map NVLink-style IPC buffer sharing to Intel GPU equivalent
      (Level Zero IPC handles: `zeMemGetIpcHandle` / `zeMemOpenIpcHandle`).
- [ ] Add shared memory allocator for Intel GPU (replace `cudaIpcMemHandle_t` /
      `CUmemFabricHandle` with L0 IPC or USM allocations).

Acceptance: `Buffer.__init__` and `Buffer.sync` complete without error on Intel GPU.

## Phase 4: SYCL Kernel Infrastructure
- [ ] Create `csrc/sycl/` directory for SYCL kernel files.
- [ ] Port common utilities from `utils.cuh` to SYCL equivalents:
  - Warp-level → sub-group level intrinsics mapping.
  - `__shared__` → `sycl::local_accessor` local memory.
  - `__syncthreads()` → `sycl::group_barrier()`.
  - `atomicCAS/atomicAdd` → `sycl::atomic_ref`.
  - `clock64()` for timeout → `sycl::ext::intel::experimental::read_cycle_counter()` or equivalent.
  - `ld_volatile_global/st_na_global` → SYCL atomic fences + memory order.
- [ ] Port `configs.cuh` constants to SYCL-compatible header (`sycl/configs.hpp`).
- [ ] Port `exception.cuh` assertion macros to SYCL device assertions.
- [ ] Port `buffer.cuh` symmetric buffer helpers to SYCL.
- [ ] Create SYCL launch config utilities (nd_range, sub_group_size selection).

Acceptance: SYCL utility headers compile with `icpx -fsycl`.

## Phase 5: SYCL iSHMEM IBGDA Device-Side Abstraction
- [ ] Create `csrc/sycl/ibgda_device.hpp` — SYCL IBGDA device API wrapper:
  - `ibgda_put_nbi_subgroup()` — maps to `ishmemi_ibgda_device_emit_direct_wqe_skeleton(PUT)`.
  - `ibgda_quiet()` — maps to CQ polling via `ishmemi_ibgda_uc_load16` or iSHMEM quiet.
  - `ibgda_amo_nonfetch_add()` — maps to iSHMEM atomic fetch-add via `ishmemi_ibgda_device_emit_direct_wqe_skeleton` with atomic opcode.
  - `get_p2p_ptr()` — maps to `ishmem_ptr()` for intra-node peer pointer resolution.
  - `barrier_all_block()` — maps to `ishmem_barrier_all()` within work-group scope.
  - `sync_all()` / `sync_team()` — maps to `ishmem_team_sync()` or `ishmem_sync_all()`.

NVSHMEM → iSHMEM IBGDA device API mapping table:

| NVSHMEM (CUDA)                        | iSHMEM (SYCL)                                        |
|---------------------------------------|------------------------------------------------------|
| `nvshmemi_ibgda_put_nbi_warp`         | `ishmemi_ibgda_device_emit_direct_wqe_skeleton(PUT)` |
| `nvshmemi_ibgda_quiet`                | CQ poll via `ishmemi_ibgda_uc_load16` + fence        |
| `nvshmemi_ibgda_amo_nonfetch_add`     | Direct WQE with `MLX5_OPCODE_ATOMIC_FA`              |
| `nvshmemi_ibgda_rma_p`               | `ishmemi_ibgda_device_emit_staged_wqe(PUT)`          |
| `nvshmemi_get_p2p_ptr`               | `ishmem_ptr()`                                        |
| `nvshmem_sync_all`                    | `ishmem_sync_all()`                                   |
| `nvshmem_sync(team)`                  | `ishmem_team_sync(team)`                              |
| `nvshmemx_barrier_all_block`          | `ishmem_barrier_all()` (work-group scope)             |
| `nvshmem_team_split_strided`          | `ishmem_team_split_strided()`                         |

Acceptance: IBGDA wrapper compiles and iSHMEM device context is accessible from SYCL kernel.

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
