# DeepEP iSHMEM + SYCL Integration Design

## 1. Background

DeepEP is a Mixture-of-Experts (MoE) expert-parallel communication library providing:
- **High-throughput internode** all-to-all dispatch/combine (RDMA + NVLink)
- **Low-latency internode** dispatch/combine (RDMA-only, IBGDA)
- **Intranode** all-to-all dispatch/combine (NVLink)

The existing implementation is tightly coupled to CUDA + NVSHMEM:
- GPU kernels are written in CUDA (`__global__`, warp intrinsics, shared memory)
- Internode communication uses NVSHMEM IBGDA device-side APIs called **directly from GPU kernels**
- Memory management uses CUDA IPC / CU mem fabric handles
- Tensor operations use PyTorch CUDA tensors

This document describes how to implement a complete **SYCL + iSHMEM** backend that
delivers equivalent DeepEP functionality on Intel GPUs.

## 2. Scope

### In Scope
- Full SYCL reimplementation of all GPU kernels (internode, low-latency, intranode, layout)
- iSHMEM IBGDA device-side integration for RDMA data movement from SYCL kernels
- iSHMEM host runtime integration (init, symmetric heap, barriers, teams)
- Intel GPU memory management (Level Zero IPC, USM)
- `torch.xpu` tensor support via Intel Extension for PyTorch

### Out of Scope
- Unified CUDA/SYCL kernel source (separate kernel files per backend)
- MNNVL (Multi-Node NVLink) equivalent for Intel — deferred
- Performance parity with NVIDIA in first release

## 3. Design Principles

1. **Separate kernel implementations**: CUDA kernels stay in `csrc/kernels/`, SYCL kernels
   go in `csrc/sycl/`. No cross-contamination.
2. **Shared host orchestration**: `deep_ep.cpp` dispatches to the correct backend's kernel
   launcher based on compile-time `#ifdef DEEPEP_USE_SYCL`.
3. **Same Python API**: `deep_ep/buffer.py` public interface is unchanged. Backend selection
   is via `DEEPEP_INTERNODE_BACKEND` env var.
4. **iSHMEM IBGDA for lowest latency**: Use GPU-direct NIC doorbell posting from SYCL kernels,
   not proxy-based fallback.

## 4. Code Structure

### 4.1 Source Tree

```
csrc/
├── kernels/                       # Existing CUDA backend (unchanged)
│   ├── internode.cu               # 2377 lines — high-throughput internode
│   ├── internode_ll.cu            # 1292 lines — low-latency internode
│   ├── intranode.cu               # 1103 lines — intranode NVLink
│   ├── layout.cu                  #  153 lines — dispatch layout
│   ├── runtime.cu                 #   98 lines — NVSHMEM host runtime
│   ├── ibgda_device.cuh           # NVSHMEM IBGDA device WQE helpers
│   ├── configs.cuh                # Constants + NVSHMEM/CUDA includes
│   ├── utils.cuh                  # GPU utility macros (ld/st, fences)
│   ├── buffer.cuh                 # Symmetric buffer template
│   ├── launch.cuh                 # CUDA launch config
│   ├── exception.cuh              # Device assertions
│   └── api.cuh                    # Kernel function declarations
│
├── sycl/                          # New SYCL backend
│   ├── configs.hpp                # Constants + iSHMEM/SYCL includes
│   ├── utils.hpp                  # Sub-group, atomic, fence utilities
│   ├── exception.hpp              # Device assertions
│   ├── buffer.hpp                 # Symmetric buffer for iSHMEM heap
│   ├── ibgda_device.hpp           # iSHMEM IBGDA device wrapper
│   ├── launch.hpp                 # SYCL launch helpers
│   ├── api.hpp                    # Kernel function declarations
│   ├── internode.cpp              # High-throughput internode kernels
│   ├── internode_ll.cpp           # Low-latency internode kernels
│   ├── intranode.cpp              # Intranode kernels
│   ├── layout.cpp                 # Layout kernels
│   └── runtime.cpp                # iSHMEM host runtime
│
├── deep_ep.cpp                    # Unified host orchestration
├── deep_ep.hpp                    # Buffer class (backend-conditional members)
├── config.hpp                     # Backend-conditional includes
├── internode_runtime_adapter.hpp  # Backend-neutral runtime API
└── event.hpp                      # Event handling
```

### 4.2 Compilation Model

| Backend        | Compiler | Source files compiled        | Key defines              |
|----------------|----------|-----------------------------|--------------------------|
| `cuda_nvshmem` | nvcc     | `csrc/kernels/*.cu`         | (default)                |
| `sycl_ishmem`  | icpx     | `csrc/sycl/*.cpp`           | `DEEPEP_USE_SYCL`, `DISABLE_NVSHMEM` |
| `none`         | g++      | `csrc/deep_ep.cpp` only     | `DISABLE_NVSHMEM`        |

## 5. CUDA → SYCL Kernel Porting Guide

### 5.1 Execution Model Mapping

| CUDA                              | SYCL                                                      |
|-----------------------------------|------------------------------------------------------------|
| `__global__ void kernel(...)`     | `q.parallel_for<class K>(nd_range, [=](nd_item item){})` |
| `blockIdx.x`                      | `item.get_group(0)`                                        |
| `threadIdx.x`                     | `item.get_local_id(0)`                                     |
| `gridDim.x`                       | `item.get_group_range(0)`                                  |
| `blockDim.x`                      | `item.get_local_range(0)`                                  |
| `warp_id = threadIdx.x / 32`     | `sg.get_group_id()`                                        |
| `lane_id = threadIdx.x % 32`     | `sg.get_local_id()`                                        |
| `__syncthreads()`                 | `sycl::group_barrier(item.get_group())`                    |
| `__shfl_sync(mask, val, src)`     | `sycl::select_from_group(sg, val, src)`                    |
| `__shared__ T arr[N]`            | `sycl::local_accessor<T, 1> arr(N, h)`                    |
| `atomicCAS(ptr, cmp, val)`        | `sycl::atomic_ref<T,...>(*ptr).compare_exchange_strong()`  |
| `atomicAdd(ptr, val)`             | `sycl::atomic_ref<T,...>(*ptr).fetch_add(val)`             |
| `atomicExch(ptr, val)`            | `sycl::atomic_ref<T,...>(*ptr).exchange(val)`              |
| `clock64()`                       | architecture-specific cycle counter or steady_clock        |
| `__threadfence()`                 | `sycl::atomic_fence(release, device)`                      |
| `__threadfence_system()`          | `sycl::atomic_fence(release, system)`                      |
| `__launch_bounds__(N, M)`         | `[[intel::reqd_sub_group_size(16)]]` + tuning              |

### 5.2 Sub-Group Size

Intel GPUs use sub-group sizes of 16 or 32 (configurable).
DeepEP warp-level code assumes warp size = 32. Options:
- Use `[[intel::reqd_sub_group_size(32)]]` where 32 is supported (e.g., PVC).
- Adapt algorithms for sub-group size 16 with explicit mapping.

Recommendation: Start with sub-group size 16 (native for Intel Xe/PVC), adapt warp-level
algorithms to use `sub_group_size` variable instead of hardcoded 32.

### 5.3 Memory Semantics

| CUDA                          | SYCL                                                                    |
|-------------------------------|-------------------------------------------------------------------------|
| `ld_volatile_global(ptr)`     | `sycl::atomic_ref<T, relaxed, system, global_space>(*ptr).load()`      |
| `st_na_global(ptr, val)`      | Normal store + release fence                                            |
| `ld_acquire_global(ptr)`      | `sycl::atomic_ref<T, acquire, device, global_space>(*ptr).load()`      |
| `st_release_sys_global(p,v)`  | `sycl::atomic_ref<T, release, system, global_space>(*p).store(v)`      |

## 6. NVSHMEM → iSHMEM IBGDA Device API Mapping

### 6.1 Host Runtime

| NVSHMEM                         | iSHMEM                                     | Notes                            |
|---------------------------------|--------------------------------------------|----------------------------------|
| `nvshmemx_get_uniqueid`         | N/A — use PMI or MPI bootstrap             | iSHMEM uses runtime bootstrap    |
| `nvshmemx_init_attr`            | `ishmem_init()` / `ishmemx_init_attr()`    | Set runtime type in attr         |
| `nvshmem_my_pe()`               | `ishmem_my_pe()`                            |                                  |
| `nvshmem_align(align, size)`    | `ishmem_align(align, size)`                 |                                  |
| `nvshmem_malloc(size)`          | `ishmem_malloc(size)`                       |                                  |
| `nvshmem_free(ptr)`             | `ishmem_free(ptr)`                          |                                  |
| `nvshmem_barrier_all()`         | `ishmem_barrier_all()`                      |                                  |
| `nvshmem_finalize()`            | `ishmem_finalize()`                         |                                  |
| `nvshmem_team_split_strided`    | `ishmem_team_split_strided`                 | Same semantics                   |
| `nvshmem_team_destroy`          | `ishmem_team_destroy`                       |                                  |

### 6.2 Device-Side IBGDA

DeepEP's CUDA kernels directly construct and post mlx5 NIC WQEs via NVSHMEM's
`ibgda_device.cuh`. The iSHMEM equivalent (`ibgda_device_impl.h`) provides the same
capability for SYCL kernels.

| DeepEP NVSHMEM call                                  | iSHMEM equivalent                                                    |
|------------------------------------------------------|----------------------------------------------------------------------|
| `nvshmemi_ibgda_put_nbi_warp<signal>(dst, src, n, pe, qp, lane, slot)` | `ishmemi_ibgda_device_emit_direct_wqe_skeleton(ctx, peer, PUT, src, dst, n, pe)` |
| `nvshmemi_ibgda_quiet(pe, qp_id)`                   | CQ poll: `ishmemi_ibgda_uc_load16(peer_ctx->nic_cq_buf + 0x3C)` until `wqe_idx` completes |
| `nvshmemi_ibgda_amo_nonfetch_add(ptr, val, pe, qp)` | Build WQE with `ISHMEMI_IBGDA_MLX5_OPCODE_ATOMIC_FA` via direct WQE emission |
| `nvshmemi_ibgda_rma_p(ptr, val, pe, qp)`            | `ishmemi_ibgda_device_emit_staged_wqe(PUT, &val, ptr, sizeof(int), pe)` or direct WQE |
| `ibgda_get_state()`                                  | `ishmemi_ibgda_device_get_context()`                                 |
| `ibgda_get_rc(pe, id)`                               | `ishmemi_ibgda_device_peer_context_qp(ctx, pe, qp_idx)`             |
| `ibgda_submit_requests(qp, base, n, msg_idx)`       | Doorbell: `ishmemi_ibgda_device_ring_doorbell(peer_ctx, wqe, prod)` |
| `ibgda_write_rdma_write_inl_wqe(...)`                | Build WQE inline in slot at `peer_ctx->nic_wq_base_addr`            |

### 6.3 Key Differences

1. **QP management**: NVSHMEM uses `ibgda_get_rc(pe, id)` to get QP by PE + RC index.
   iSHMEM uses `ishmemi_ibgda_device_peer_context_qp(ctx, pe, qp_idx)` — same concept,
   different structure layout.

2. **WQE construction**: NVSHMEM has `ibgda_write_rdma_write_inl_wqe` helpers that build
   mlx5 WQE with inline data. iSHMEM's `emit_direct_wqe_skeleton` builds a standard
   RDMA WRITE/READ WQE (ctrl + rdma + data segments). For inline data, extend with
   `MLX5_INLINE_SEG` similarly.

3. **Doorbell batching**: NVSHMEM batches via `ibgda_submit_requests` (tracks ready_head,
   posts when batch full). iSHMEM supports `db_batch_size` — only rings UAR MMIO on
   batch boundaries, DBR is always updated.

4. **Completion (quiet)**: NVSHMEM's `nvshmemi_ibgda_quiet` polls internal CQ.
   iSHMEM exposes `nic_cq_buf` directly — poll CQE `wqe_counter` with
   `ishmemi_ibgda_uc_load16` (uncacheable load).

5. **Address translation**: NVSHMEM uses internal VA mapping. iSHMEM uses
   `peer_ctx->remote_vaddr + (local_ptr - ctx->heap_base)` for remote address
   computation within the symmetric heap.

6. **Sub-group vs warp**: `nvshmemi_ibgda_put_nbi_warp` is warp-cooperative (32 lanes
   participate). The SYCL equivalent needs to be sub-group-cooperative with configurable
   sub-group size.

## 7. Internode Data Flow

### 7.1 High-Throughput Dispatch (internode.cu → sycl/internode.cpp)

```
Phase 1: Metadata exchange
  ┌─────────────────────────────────────────────────────┐
  │ 1. Quiet all in-flight IBGDA operations              │
  │ 2. Intra-node barrier (NVLink/P2P)                   │
  │ 3. Inter-node barrier (ishmem_sync or team_sync)     │
  │ 4. IBGDA put: num_tokens_per_rank to all RDMA peers  │
  │ 5. Quiet + barrier                                    │
  └─────────────────────────────────────────────────────┘
         ↓
Phase 2: Data transfer
  ┌─────────────────────────────────────────────────────┐
  │ 1. Pack token data (hidden, scales, topk) per channel │
  │ 2. For each destination RDMA rank:                    │
  │    - IBGDA put_nbi: transfer packed token data        │
  │    - IBGDA atomic_add: update tail counter            │
  │ 3. Consumer polls head counters for arrival           │
  └─────────────────────────────────────────────────────┘
```

### 7.2 Low-Latency Dispatch (internode_ll.cu → sycl/internode_ll.cpp)

```
Send phase:
  ┌─────────────────────────────────────────────────────┐
  │ For each token assigned to a remote expert:           │
  │ 1. FP8 quantize (if enabled)                          │
  │ 2. Pack: [src_info | hidden_data | scales]            │
  │ 3. IBGDA put_nbi to dst PE at expert's buffer slot    │
  │ 4. IBGDA atomic_add to signal dst expert counter      │
  └─────────────────────────────────────────────────────┘

Recv phase:
  ┌─────────────────────────────────────────────────────┐
  │ For each local expert:                                │
  │ 1. Poll atomic counter for expected token count       │
  │ 2. Unpack received tokens from RDMA buffer            │
  │ 3. Record src_info and layout_range for combine       │
  └─────────────────────────────────────────────────────┘
```

## 8. Memory Model

### 8.1 Buffer Types

| Buffer            | NVSHMEM (existing)        | iSHMEM (new)                    |
|-------------------|---------------------------|---------------------------------|
| RDMA buffer       | `nvshmem_align()`         | `ishmem_align()`                |
| NVL buffer        | `cudaMalloc` + IPC handle | L0 USM + `zeMemGetIpcHandle`    |
| Barrier signals   | `cudaMalloc` + IPC handle | L0 USM + IPC handle             |
| Workspace         | `cudaMalloc`              | `sycl::malloc_device`           |
| Host-mapped       | `cudaHostAlloc` + map     | `sycl::malloc_host` + map       |

### 8.2 Symmetric Heap

iSHMEM symmetric heap semantics are identical to NVSHMEM:
- `ishmem_align(alignment, size)` allocates from the symmetric heap
- All PEs get the same virtual address offset
- IBGDA WQEs reference remote data via `peer_ctx->remote_vaddr + (ptr - heap_base)`

### 8.3 Intranode P2P

DeepEP uses CUDA IPC memory handles for NVLink P2P between GPUs on the same node.
For Intel GPU:
- Use Level Zero IPC: `zeMemGetIpcHandle()` / `zeMemOpenIpcHandle()`
- Or use `ishmem_ptr()` for peer pointer resolution within the symmetric heap

## 9. FP8 Support

DeepEP uses `__nv_fp8_storage_t` / `__nv_fp8_e4m3` (CUDA `cuda_fp8.h`).
For Intel GPU:
- Intel Xe/PVC does not have native FP8 hardware support in the same way
- Implement FP8 as `uint8_t` storage with manual conversion:
  - E4M3: same bit layout as NVIDIA, manual float↔FP8 conversion
  - UE8M0: exponent-only format, manual conversion
- Use SYCL `sycl::half` and `sycl::ext::oneapi::bfloat16` for BF16

## 10. PyTorch Integration

### 10.1 Tensor Device

| CUDA                     | Intel GPU                              |
|--------------------------|----------------------------------------|
| `torch.cuda` tensors     | `torch.xpu` tensors (via IPEX)         |
| `at::cuda::CUDAStream`  | `sycl::queue` (wrapped)                |
| `cudaStream_t`            | `sycl::queue*`                          |
| `cudaEvent_t`             | `sycl::event`                           |
| `AT_CUDA_CHECK`           | Check L0/SYCL error codes               |

### 10.2 Extension Build

CUDA: `torch.utils.cpp_extension.CUDAExtension`
SYCL: Custom build via `setuptools.Extension` + `icpx -fsycl` flags, or use
Intel Extension for PyTorch build infrastructure.

## 11. Error Handling and Diagnostics

- All SYCL kernel submissions check for `sycl::exception`.
- iSHMEM errors are checked at host runtime call sites.
- Device-side assertions use SYCL `assert()` (available with `-DSYCL_DEVICE_ONLY`).
- Timeout detection: replace `clock64()` with architecture-appropriate cycle counter.
- Mask buffer for unhealthy peers: same logic, SYCL atomic operations.

## 12. Validation Plan

### 12.1 Correctness
- Bit-exact comparison of dispatch/combine output between CUDA and SYCL backends
  using identical input tensors.
- Multi-PE (2-node, 4-node) internode tests.
- Edge cases: zero tokens to some ranks, all tokens to one rank.

### 12.2 Functional Parity
- FP8 + BF16 mixed precision paths.
- Low-latency mode with timeout/mask.
- Cached dispatch layout path.
- Shrink mode.

### 12.3 Performance
- Latency: single token dispatch/combine round-trip.
- Throughput: maximum tokens/sec at various hidden sizes.
- IBGDA path efficiency: measure WQE posting rate and CQ drain rate.

## 13. Risk and Mitigation

| Risk                                          | Mitigation                                              |
|-----------------------------------------------|--------------------------------------------------------|
| Sub-group size mismatch (16 vs 32)            | Parameterize warp-level code, test both sizes           |
| iSHMEM IBGDA direct doorbell not yet stable   | Fall back to staged WQE + proxy path                    |
| FP8 precision differences                     | Manual conversion with reference test vectors           |
| Intel GPU L0 IPC limitations                  | Fall back to iSHMEM symmetric heap for all buffers      |
| PyTorch XPU integration maturity              | Test with latest IPEX nightly                           |

## 14. Open Questions

1. **Sub-group size strategy**: Should we mandate 32-wide sub-groups on PVC or adapt all
   algorithms to work with 16?
2. **CQ polling granularity**: iSHMEM exposes per-peer CQ buffer — is this sufficient for
   the multi-QP quiet semantics DeepEP needs?
3. **Inline WQE support**: DeepEP's `ibgda_write_rdma_write_inl_wqe` puts small payloads
   inline in the WQE. Does iSHMEM's direct WQE path support this?
4. **MNNVL equivalent**: For multi-node NVLink, is there an Intel equivalent or is RDMA-only
   sufficient for first release?
5. **Atomic ordering**: DeepEP relies on RDMA atomic add for signaling. Verify iSHMEM
   atomic FA WQE provides same ordering guarantees through the NIC.
