# DeepEP iSHMEM Integration Phase 0 Baseline

## Purpose
This document defines the Phase 0 baseline for DeepEP iSHMEM integration:
- backend terminology normalization
- target hardware/software matrix
- scope boundary for Phase 0 completion

## Backend Terminology
- `cuda_nvshmem`: CUDA stack with NVSHMEM-based internode communication.
- `sycl_ishmem`: SYCL stack with iSHMEM-based internode communication.

These terms are planning and code-level identifiers and should be used consistently
in docs, code comments, TODO tracking, and test naming.

## Target Hardware and Software Matrix

| Backend | Target GPU Vendor | Device Programming Stack | Communication Runtime | Internode Path |
|---|---|---|---|---|
| `cuda_nvshmem` | NVIDIA | CUDA | NVSHMEM | NVSHMEM (existing DeepEP path) |
| `sycl_ishmem` | Intel | SYCL (DPC++) | iSHMEM | iSHMEM IBGDA-capable path |

## Phase 0 Completion Criteria
- `doc/` planning documents are present.
- backend terminology is documented and exported at package entry.
- target hardware/software matrix is documented.

## Notes
Phase 0 does not include:
- SYCL kernel porting for dispatch/combine.
- runtime feature parity with existing CUDA path.
- low-latency implementation parity.

Those are covered by later phases in `doc/TODO.md`.
