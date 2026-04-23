# DeepEP iSHMEM Integration — Baseline Reference

## Purpose
Baseline facts for the DeepEP iSHMEM + SYCL backend integration.

## Backend Terminology
- `cuda_nvshmem`: CUDA stack with NVSHMEM-based internode communication (existing).
- `sycl_ishmem`: SYCL stack with iSHMEM-based internode communication (target).

## Target Hardware and Software Matrix

| Backend | GPU Vendor | Device Stack | Communication | IBGDA | NIC |
|---|---|---|---|---|---|
| `cuda_nvshmem` | NVIDIA | CUDA | NVSHMEM | NVSHMEM ibgda_device | ConnectX (mlx5) |
| `sycl_ishmem` | Intel | SYCL (DPC++) | iSHMEM | iSHMEM ibgda_device_impl | ConnectX (mlx5) / bnxt_re |

## DeepEP Kernel Inventory

| Kernel file        | Lines | Role                               | NVSHMEM IBGDA usage          |
|--------------------|-------|------------------------------------|------------------------------|
| `internode.cu`     | 2377  | High-throughput internode dispatch/combine | put_nbi_warp, quiet, amo, sync |
| `internode_ll.cu`  | 1292  | Low-latency dispatch/combine       | put_nbi_warp, quiet, amo, rma_p, barrier |
| `intranode.cu`     | 1103  | NVLink intranode dispatch/combine  | None (NVLink P2P only)       |
| `layout.cu`        |  153  | Dispatch layout computation        | None                         |
| `runtime.cu`       |   98  | Host runtime (init/alloc/barrier)  | Host API only                |

## NVSHMEM IBGDA Device APIs Used in DeepEP

- `nvshmemi_ibgda_put_nbi_warp<signal>(dst, src, n, pe, qp, lane, slot)` — warp-cooperative RDMA WRITE
- `nvshmemi_ibgda_quiet(pe, qp_id)` — drain all pending WQEs for a QP
- `nvshmemi_ibgda_amo_nonfetch_add(ptr, val, pe, qp)` — atomic fetch-add via NIC
- `nvshmemi_ibgda_rma_p(ptr, val, pe, qp)` — single-element put (inline WQE)
- `ibgda_get_state()` — get global IBGDA device state
- `ibgda_get_rc(pe, id)` — get RC QP descriptor for a PE
- `ibgda_submit_requests(qp, base, n, msg_idx)` — commit WQEs + doorbell
- `ibgda_write_rdma_write_inl_wqe(...)` — build inline RDMA WRITE WQE
- `nvshmemi_get_p2p_ptr(addr, src, dst)` — resolve P2P pointer
- `nvshmem_sync_all()` / `nvshmem_sync(team)` — global/team sync
- `nvshmemx_barrier_all_block()` — block-scope barrier

## iSHMEM IBGDA Device APIs Available

- `ishmemi_ibgda_device_get_context()` — get global IBGDA device context
- `ishmemi_ibgda_device_peer_context(ctx, pe)` — get peer context (default QP)
- `ishmemi_ibgda_device_peer_context_qp(ctx, pe, qp_idx)` — get peer context (specific QP)
- `ishmemi_ibgda_device_emit_staged_wqe(op, local, remote, n, pe)` — proxy-assisted WQE
- `ishmemi_ibgda_device_emit_direct_wqe_skeleton(ctx, peer, op, local, remote, n, pe)` — direct NIC doorbell
- `ishmemi_ibgda_device_emit_direct_wqe_blocking(ctx, peer, op, local, remote, n, pe)` — blocking variant
- `ishmemi_ibgda_device_ring_doorbell(peer, wqe, prod)` — ring UAR MMIO
- `ishmemi_ibgda_device_can_direct_doorbell(ctx)` — check readiness
- `ishmemi_ibgda_uc_load16(ptr)` — uncacheable 16-bit load (CQ polling)
- `ishmemi_ibgda_htobe32/64(x)` — byte-swap helpers

## Phase 0 Completion
- [x] Backend terminology defined
- [x] Target matrix documented
- [x] Kernel inventory documented
- [x] NVSHMEM ↔ iSHMEM IBGDA API surface mapped
