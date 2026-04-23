#pragma once

// iSHMEM host runtime wrapper for DeepEP SYCL backend.
// Encapsulates ishmem_init / malloc / free / barrier / finalize
// and provides the same interface contract as NVSHMEM in the CUDA path.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef SYCL_ISHMEM
#include <ishmem.h>
#include <ishmemx.h>
#endif

#include "sycl_backend/exception.hpp"

namespace deep_ep {
namespace ishmem_runtime {

// ---------------------------------------------------------------------------
// Initialization / finalize
// ---------------------------------------------------------------------------

// Initialize iSHMEM. Must be called before any other iSHMEM operation.
// Returns the PE (processing element) index, which corresponds to the RDMA rank.
inline int init() {
#ifdef SYCL_ISHMEM
    ishmem_init();
    int pe = ishmem_my_pe();
    int npes = ishmem_n_pes();
    printf("DeepEP iSHMEM: initialized PE %d / %d\n", pe, npes);
    return pe;
#else
    EP_HOST_ASSERT(false and "iSHMEM not available (SYCL_ISHMEM not defined)");
    return -1;
#endif
}

// Initialize iSHMEM with attributes (e.g., selecting MPI/PMI runtime).
inline int init_attr(bool use_mpi = false) {
#ifdef SYCL_ISHMEM
    ishmemx_attr_t attr;
    attr.runtime = use_mpi ? ISHMEMX_RUNTIME_MPI : ISHMEMX_RUNTIME_PMI;
    attr.initialize_runtime = true;
    attr.gpu = true;
    attr.mpi_comm = nullptr;
    ishmemx_init_attr(&attr);
    int pe = ishmem_my_pe();
    int npes = ishmem_n_pes();
    printf("DeepEP iSHMEM: initialized PE %d / %d (runtime=%s)\n",
           pe, npes, use_mpi ? "MPI" : "PMI");
    return pe;
#else
    EP_HOST_ASSERT(false and "iSHMEM not available (SYCL_ISHMEM not defined)");
    return -1;
#endif
}

inline int my_pe() {
#ifdef SYCL_ISHMEM
    return ishmem_my_pe();
#else
    return -1;
#endif
}

inline int n_pes() {
#ifdef SYCL_ISHMEM
    return ishmem_n_pes();
#else
    return 0;
#endif
}

inline void finalize() {
#ifdef SYCL_ISHMEM
    ishmem_finalize();
#endif
}

// ---------------------------------------------------------------------------
// Symmetric heap memory management
// ---------------------------------------------------------------------------

// Allocate from the iSHMEM symmetric heap (remotely accessible by all PEs).
inline void* alloc(size_t size, size_t alignment) {
#ifdef SYCL_ISHMEM
    void* ptr = ishmem_align(alignment, size);
    EP_HOST_ASSERT(ptr != nullptr and "ishmem_align failed — symmetric heap exhausted?");
    return ptr;
#else
    EP_HOST_ASSERT(false and "iSHMEM not available");
    return nullptr;
#endif
}

inline void* malloc(size_t size) {
#ifdef SYCL_ISHMEM
    void* ptr = ishmem_malloc(size);
    EP_HOST_ASSERT(ptr != nullptr and "ishmem_malloc failed");
    return ptr;
#else
    EP_HOST_ASSERT(false and "iSHMEM not available");
    return nullptr;
#endif
}

inline void free(void* ptr) {
#ifdef SYCL_ISHMEM
    if (ptr) ishmem_free(ptr);
#endif
}

// ---------------------------------------------------------------------------
// Synchronization
// ---------------------------------------------------------------------------

inline void barrier_all() {
#ifdef SYCL_ISHMEM
    ishmem_barrier_all();
#endif
}

inline void sync_all() {
#ifdef SYCL_ISHMEM
    ishmem_sync_all();
#endif
}

// ---------------------------------------------------------------------------
// Teams (sub-groups of PEs)
// ---------------------------------------------------------------------------

#ifdef SYCL_ISHMEM
inline int team_split_strided(ishmem_team_t parent_team,
                              int start, int stride, int size,
                              const ishmem_team_config_t* config,
                              long config_mask,
                              ishmem_team_t* new_team) {
    return ishmem_team_split_strided(parent_team, start, stride, size,
                                     config, config_mask, new_team);
}
#endif

// ---------------------------------------------------------------------------
// Peer pointer resolution
// ---------------------------------------------------------------------------

// Get a locally-addressable pointer to a symmetric object on a remote PE.
// Returns nullptr if not directly accessible (i.e., not on the same node).
inline void* ptr(const void* dest, int pe) {
#ifdef SYCL_ISHMEM
    return ishmem_ptr(dest, pe);
#else
    return nullptr;
#endif
}

}  // namespace ishmem_runtime
}  // namespace deep_ep
