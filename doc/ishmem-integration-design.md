# DeepEP iSHMEM Integration Design

## 1. Background
DeepEP currently uses CUDA kernels and NVSHMEM-centric internode communication behavior.
For Intel GPU targets, DeepEP needs a SYCL + iSHMEM backend. This design describes
how to integrate iSHMEM (including IBGDA-capable path) without regressing the existing
CUDA/NVSHMEM backend.

## 2. Scope
In scope:
- Backend abstraction for internode runtime and communication orchestration.
- New SYCL+iSHMEM backend path for internode dispatch/combine.
- Low-latency mode mapping to iSHMEM APIs and completion semantics.

Out of scope (initially):
- Full kernel-level feature parity in first merge.
- Cross-vendor unified device kernel source in one codepath.

## 3. Design Principles
- Preserve existing CUDA/NVSHMEM behavior by default.
- Additive architecture: introduce `sycl_ishmem` backend in parallel.
- Clear capability contracts: runtime can query supported features.
- Fail-fast diagnostics for unsupported combinations.

## 4. High-Level Architecture
### 4.1 Backend Layering
- Python API layer (`deep_ep/buffer.py`) remains user entry.
- Runtime adapter chooses backend implementation.
- Backend implementation owns:
  - init/finalize
  - rank/team mapping
  - buffer allocation and lifecycle
  - dispatch/combine internode communication

### 4.2 Suggested Interfaces
Backend-neutral internode runtime interface should expose:
- `get_unique_id()`
- `init(root_id, rank, world_size, mode)`
- `alloc(size, alignment)`
- `free(ptr)`
- `barrier()`
- `finalize()`

Backend capability interface should expose:
- `supports_internode`
- `supports_low_latency`
- `supports_fp8_fast_path`

## 5. Data Path Mapping (Conceptual)
### 5.1 Dispatch
- Prepare token layout and per-rank metadata.
- Use iSHMEM RMA/NBI operations for transfer.
- Use backend barrier/quiet semantics to ensure visibility and completion.

### 5.2 Combine
- Read remote contributions with deterministic ordering constraints.
- Apply reduction/accumulation semantics consistent with current DeepEP behavior.

### 5.3 Low Latency
- Use iSHMEM IBGDA-capable primitives where available.
- Keep protocol invariants:
  - bounded in-flight messages
  - explicit completion points
  - timeout/mask handling for unhealthy peers

## 6. Rank and Team Model
- Global PE index maps to distributed rank.
- Local device grouping and internode grouping are backend-defined.
- Team split strategy should be explicit and deterministic.

## 7. Memory Model
- Communication buffers must be backend-accessible symmetric memory.
- Alignment and granularity constraints are backend-specific.
- Buffer ownership and cleanup must be deterministic across failures.

## 8. Error Handling and Diagnostics
- Distinguish configuration errors, runtime errors, and unsupported features.
- Include backend name and required environment in error messages.
- Add optional verbose logging for backend init and communication progress.

## 9. Validation Plan
### 9.1 Correctness
- Bitwise/relative tolerance checks vs reference implementation.
- Multi-rank dispatch/combine consistency tests.

### 9.2 Reliability
- Timeout/mask behavior under delayed or failed peers.
- Repeated init/finalize and buffer lifecycle stress tests.

### 9.3 Performance
- Throughput and latency benchmarks by message size and rank count.
- Comparison against baseline backend on supported platforms.

## 10. Rollout Plan
- Step 1: Land backend abstraction and compile-time switches.
- Step 2: Land minimal SYCL+iSHMEM internode dispatch/combine path.
- Step 3: Land low-latency protocol support and parity tests.
- Step 4: Optimize performance and finalize docs.

## 11. Open Questions
- Exact API subset of iSHMEM for lowest-latency path in DeepEP workflow.
- Preferred team split policy for heterogeneous cluster layouts.
- FP8 path strategy for Intel GPU first release.
