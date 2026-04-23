# DeepEP iSHMEM Integration TODO

## Goals
- Add an Intel GPU (SYCL + iSHMEM) backend path in DeepEP.
- Keep existing CUDA + NVSHMEM path unchanged.
- Deliver a functional internode dispatch/combine implementation with iSHMEM IBGDA support.

## Phase 0: Project Setup
- [x] Add `doc/` docs and planning artifacts.
- [x] Define backend terminology: `cuda_nvshmem`, `sycl_ishmem`.
- [x] Document target hardware/software matrix.

Status:
- Completed on 2026-04-23.
- Baseline references: `doc/phase0-baseline.md`, `doc/ishmem-integration-design.md`.

## Phase 1: Backend Abstraction
- [ ] Introduce backend-neutral internode runtime API (init, alloc, barrier, finalize, rank info).
- [ ] Isolate CUDA-specific internode calls behind adapter interfaces.
- [ ] Add capability flags in Python runtime (`supports_low_latency`, `supports_fp8_path`, etc.).

## Phase 2: Build and Packaging
- [ ] Add SYCL build options and dependency checks for iSHMEM.
- [ ] Add a build-time selector for backend and architecture.
- [ ] Ensure wheel/source build paths fail fast with actionable diagnostics.

## Phase 3: Runtime Integration
- [ ] Add iSHMEM initialization path in DeepEP runtime wrapper.
- [ ] Add team/rank mapping policy for internode and local-group decomposition.
- [ ] Add symmetric memory allocation strategy for recv/send buffers.

## Phase 4: Data Path (Minimal Working Version)
- [ ] Implement minimal internode dispatch data movement via iSHMEM RMA/NBI APIs.
- [ ] Implement minimal internode combine return path.
- [ ] Add correctness checks against reference tensors.

## Phase 5: Low-Latency Path
- [ ] Map low-latency send/recv protocol to iSHMEM IBGDA-capable APIs.
- [ ] Implement synchronization and completion semantics equivalent to existing behavior.
- [ ] Add timeout/mask behavior parity tests.

## Phase 6: Performance and Feature Parity
- [ ] Optimize chunking, queueing, and overlap strategy.
- [ ] Validate throughput and latency against target baselines.
- [ ] Evaluate FP8/scale handling parity and memory overhead.

## Phase 7: Testing and CI
- [ ] Add backend-specific unit/integration tests.
- [ ] Add multi-node regression tests for dispatch/combine.
- [ ] Add CI job matrix with backend toggles and sanity checks.

## Deliverables
- [ ] Design doc approved.
- [ ] Minimal iSHMEM backend merged.
- [ ] Low-latency parity merged.
- [ ] Benchmarks and migration notes published.
