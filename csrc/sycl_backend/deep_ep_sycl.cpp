// DeepEP SYCL + iSHMEM backend entry point
// This file provides the same pybind11 module interface as deep_ep.cpp (CUDA backend),
// but targets Intel GPUs via SYCL and iSHMEM for internode communication.
//
// Phase 3: iSHMEM host runtime integration — real Buffer init/sync/destroy.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <torch/types.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "sycl_backend/configs.hpp"
#include "sycl_backend/exception.hpp"
#include "sycl_backend/sycl_context.hpp"
#include "sycl_backend/l0_ipc_memory.hpp"
#include "sycl_backend/ishmem_runtime.hpp"
#include "sycl_backend/sycl_utils.hpp"
#include "sycl_backend/sycl_buffer.hpp"
#include "sycl_backend/sycl_launch.hpp"
#include "sycl_backend/ibgda_device.hpp"

namespace py = pybind11;

namespace deep_ep {

// ---------------------------------------------------------------------------
// Config — mirrors the CUDA Config struct
// ---------------------------------------------------------------------------
struct Config {
    int num_sms;
    int num_max_p2p_chunked_send_tokens;
    int num_max_p2p_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;

    Config(int num_sms = 20,
           int num_max_p2p_chunked_send_tokens = 6,
           int num_max_p2p_chunked_recv_tokens = 256,
           int num_max_rdma_chunked_send_tokens = 6,
           int num_max_rdma_chunked_recv_tokens = 256)
        : num_sms(num_sms),
          num_max_p2p_chunked_send_tokens(num_max_p2p_chunked_send_tokens),
          num_max_p2p_chunked_recv_tokens(num_max_p2p_chunked_recv_tokens),
          num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
          num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {}

    int64_t get_p2p_buffer_size_hint(int num_tokens, int hidden, int num_experts, int num_ranks,
                                     int num_topk = 1) const {
        // Placeholder — will be implemented with real layout math in later phases
        return static_cast<int64_t>(num_tokens) * hidden * 2 + 1024 * 1024;
    }

    int64_t get_rdma_buffer_size_hint(int num_tokens, int hidden, int num_experts, int num_ranks,
                                      int num_topk = 1) const {
        return static_cast<int64_t>(num_tokens) * hidden * 2 + 1024 * 1024;
    }
};

int64_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank,
                                       int hidden,
                                       int num_experts,
                                       int num_ranks) {
    return static_cast<int64_t>(num_max_dispatch_tokens_per_rank) * hidden * 4 * num_ranks + 1024 * 1024;
}

// ---------------------------------------------------------------------------
// EventHandle — SYCL event wrapper (stub)
// ---------------------------------------------------------------------------
struct EventHandle {
    EventHandle() {}
    void current_stream_wait() const {
        // TODO(phase-10): implement with sycl::event
    }
};

// ---------------------------------------------------------------------------
// Buffer — main communication buffer for SYCL + iSHMEM backend
// ---------------------------------------------------------------------------
struct Buffer {
    EP_STATIC_ASSERT(NUM_MAX_P2P_PEERS == 8, "The number of maximum P2P peers must be 8");

private:
    // Rank topology (mirrors CUDA backend layout)
    int rank, rdma_rank, p2p_rank;
    int num_ranks, num_rdma_ranks, num_p2p_ranks;
    int64_t num_p2p_bytes;
    int64_t num_rdma_bytes;
    bool low_latency_mode;
    bool explicitly_destroy;
    bool enable_shrink;
    bool available = false;
    bool destroyed = false;

    // Device info
    int device_id = 0;
    int num_device_eus = 0;

    // SYCL context (non-owning reference to singleton)
    SyclContext* sycl_ctx = nullptr;

    // Intranode P2P buffers (L0 IPC)
    void* buffer_ptrs[NUM_MAX_P2P_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;
    int* barrier_signal_ptrs[NUM_MAX_P2P_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu = nullptr;
    L0IpcMemHandle ipc_handles[NUM_MAX_P2P_PEERS];
    std::unique_ptr<L0MemoryAllocator> mem_allocator;

    // Internode RDMA buffer (iSHMEM symmetric heap)
    void* rdma_buffer_ptr = nullptr;

    // Shrink mode buffers (iSHMEM symmetric heap)
    int* mask_buffer_ptr = nullptr;
    int* sync_buffer_ptr = nullptr;

    // Workspace
    void* workspace = nullptr;

    // Host-mapped MoE counters
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;
    volatile int* moe_recv_expert_counter = nullptr;
    int* moe_recv_expert_counter_mapped = nullptr;
    volatile int* moe_recv_rdma_counter = nullptr;
    int* moe_recv_rdma_counter_mapped = nullptr;

    // Host allocator for pinned memory
    std::unique_ptr<L0HostAllocator> host_allocator;

    // iSHMEM initialized flag
    bool ishmem_initialized = false;

    // Low-latency buffer index
    int low_latency_buffer_idx = 0;

public:
    Buffer(int rank,
           int num_ranks,
           int64_t num_p2p_bytes,
           int64_t num_rdma_bytes,
           bool low_latency_mode,
           bool explicitly_destroy,
           bool enable_shrink,
           bool use_fabric)
        : rank(rank),
          num_ranks(num_ranks),
          num_p2p_bytes(num_p2p_bytes),
          num_rdma_bytes(num_rdma_bytes),
          low_latency_mode(low_latency_mode),
          explicitly_destroy(explicitly_destroy),
          enable_shrink(enable_shrink) {
        // Compute rank topology
        rdma_rank = rank / NUM_MAX_P2P_PEERS;
        p2p_rank = rank % NUM_MAX_P2P_PEERS;
        num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_P2P_PEERS);
        num_p2p_ranks = std::min(num_ranks, NUM_MAX_P2P_PEERS);

        // Common checks (matching CUDA backend)
        EP_HOST_ASSERT(num_p2p_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                       (num_p2p_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
        EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                       (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
        EP_HOST_ASSERT(0 <= rank and rank < num_ranks and
                       (num_ranks <= NUM_MAX_P2P_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
        EP_HOST_ASSERT(num_ranks < NUM_MAX_P2P_PEERS or num_ranks % NUM_MAX_P2P_PEERS == 0);
        if (num_rdma_bytes > 0)
            EP_HOST_ASSERT(num_ranks > NUM_MAX_P2P_PEERS or low_latency_mode);

        // Initialize SYCL context
        sycl_ctx = &SyclContext::get();
        device_id = sycl_ctx->ordinal();
        num_device_eus = sycl_ctx->num_eus();

        // Create allocators
        mem_allocator = std::make_unique<L0MemoryAllocator>(*sycl_ctx);
        host_allocator = std::make_unique<L0HostAllocator>(*sycl_ctx);

        // Metadata sizes
        int64_t barrier_signal_bytes = NUM_MAX_P2P_PEERS * sizeof(int);
        int64_t buffer_ptr_bytes = NUM_MAX_P2P_PEERS * sizeof(void*);
        int64_t barrier_signal_ptr_bytes = NUM_MAX_P2P_PEERS * sizeof(int*);

        // Allocate intranode P2P buffers
        if (num_p2p_bytes > 0) {
            size_t total_p2p = num_p2p_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes;
            mem_allocator->malloc(&buffer_ptrs[p2p_rank], total_p2p);
            mem_allocator->get_ipc_handle(&ipc_handles[p2p_rank], buffer_ptrs[p2p_rank], total_p2p);

            buffer_ptrs_gpu = reinterpret_cast<void**>(
                static_cast<uint8_t*>(buffer_ptrs[p2p_rank]) + num_p2p_bytes + barrier_signal_bytes);
            barrier_signal_ptrs[p2p_rank] = reinterpret_cast<int*>(
                static_cast<uint8_t*>(buffer_ptrs[p2p_rank]) + num_p2p_bytes);
            barrier_signal_ptrs_gpu = reinterpret_cast<int**>(
                static_cast<uint8_t*>(buffer_ptrs[p2p_rank]) + num_p2p_bytes + barrier_signal_bytes + buffer_ptr_bytes);

            // Zero barrier signals
            mem_allocator->memset_async(barrier_signal_ptrs[p2p_rank], 0, barrier_signal_bytes,
                                        sycl_ctx->comm_queue());
        }

        // Allocate workspace (32 MiB)
        mem_allocator->malloc(&workspace, NUM_WORKSPACE_BYTES);
        mem_allocator->memset_async(workspace, 0, NUM_WORKSPACE_BYTES, sycl_ctx->comm_queue());

        // Allocate host-mapped MoE counters
        moe_recv_counter = reinterpret_cast<volatile int*>(host_allocator->malloc_host(sizeof(int64_t)));
        moe_recv_counter_mapped = const_cast<int*>(moe_recv_counter);
        *moe_recv_counter = -1;

        moe_recv_expert_counter = reinterpret_cast<volatile int*>(
            host_allocator->malloc_host(sizeof(int) * NUM_MAX_LOCAL_EXPERTS));
        moe_recv_expert_counter_mapped = const_cast<int*>(moe_recv_expert_counter);
        for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
            moe_recv_expert_counter[i] = -1;

        if (num_rdma_ranks > 0) {
            moe_recv_rdma_counter = reinterpret_cast<volatile int*>(
                host_allocator->malloc_host(sizeof(int)));
            moe_recv_rdma_counter_mapped = const_cast<int*>(moe_recv_rdma_counter);
            *moe_recv_rdma_counter = -1;
        }

        // Wait for async memsets to complete
        sycl_ctx->comm_queue().wait();
    }

    ~Buffer() noexcept(false) {
        if (not explicitly_destroy) {
            destroy();
        } else if (not destroyed) {
            printf("WARNING: destroy() was not called before DeepEP buffer destruction.\n");
            fflush(stdout);
        }
    }

    bool is_available() const { return available; }

    bool is_internode_available() const {
        return is_available() and num_ranks > NUM_MAX_P2P_PEERS;
    }

    int get_num_rdma_ranks() const { return num_rdma_ranks; }

    int get_rdma_rank() const { return rdma_rank; }

    int get_root_rdma_rank(bool global) const { return global ? p2p_rank : 0; }

    int get_local_device_id() const { return device_id; }

    py::bytearray get_local_ipc_handle() const {
        const L0IpcMemHandle& handle = ipc_handles[p2p_rank];
        return py::bytearray(reinterpret_cast<const char*>(&handle), sizeof(handle));
    }

    py::bytearray get_local_internode_unique_id() const {
        // For iSHMEM, there is no "unique ID" exchange like NVSHMEM.
        // iSHMEM uses the PMI/MPI runtime for bootstrap.
        // We return a dummy identifier containing the PE rank.
        // The actual initialization is done via ishmem_init() in sync().
        int pe_rank = rank;
        std::vector<uint8_t> id(128, 0);
        std::memcpy(id.data(), &pe_rank, sizeof(pe_rank));
        return py::bytearray(reinterpret_cast<const char*>(id.data()), id.size());
    }

    py::bytearray get_local_nvshmem_unique_id() const {
        return get_local_internode_unique_id();
    }

    torch::Tensor get_local_buffer_tensor(const py::object& dtype, int64_t offset, bool use_rdma_buffer) const {
        // TODO(phase-10): implement with proper torch.xpu tensor from device pointer
        throw std::runtime_error("SYCL backend: get_local_buffer_tensor not yet implemented (phase-10)");
    }

    torch::Stream get_comm_stream() const {
        // TODO(phase-10): wrap sycl::queue as torch::Stream when IPEX is available
        throw std::runtime_error("SYCL backend: get_comm_stream not yet implemented (phase-10)");
    }

    // -----------------------------------------------------------------------
    // sync — open IPC handles, initialize iSHMEM, allocate RDMA buffer
    // -----------------------------------------------------------------------
    void sync(const std::vector<int>& device_ids,
              const std::vector<std::optional<py::bytearray>>& all_gathered_handles,
              const std::optional<py::bytearray>& root_unique_id_opt) {
        EP_HOST_ASSERT(not is_available());

        // Sync intranode IPC handles
        if (num_p2p_bytes > 0) {
            EP_HOST_ASSERT(static_cast<int>(device_ids.size()) == num_ranks);
            EP_HOST_ASSERT(static_cast<int>(all_gathered_handles.size()) == num_ranks);

            for (int i = 0, offset = rdma_rank * num_p2p_ranks; i < num_p2p_ranks; ++i) {
                EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
                auto handle_str = std::string(all_gathered_handles[offset + i].value());
                EP_HOST_ASSERT(handle_str.size() == L0_IPC_HANDLE_SIZE);

                if (offset + i != rank) {
                    std::memcpy(&ipc_handles[i], handle_str.c_str(), L0_IPC_HANDLE_SIZE);
                    mem_allocator->open_ipc_handle(&buffer_ptrs[i], &ipc_handles[i]);
                    barrier_signal_ptrs[i] = reinterpret_cast<int*>(
                        static_cast<uint8_t*>(buffer_ptrs[i]) + num_p2p_bytes);
                } else {
                    EP_HOST_ASSERT(
                        std::memcmp(&ipc_handles[i], handle_str.c_str(), L0_IPC_HANDLE_SIZE) == 0);
                }
            }

            // Copy buffer and barrier signal pointers to device-accessible memory
            sycl_ctx->comm_queue().memcpy(buffer_ptrs_gpu, buffer_ptrs,
                                           sizeof(void*) * NUM_MAX_P2P_PEERS);
            sycl_ctx->comm_queue().memcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs,
                                           sizeof(int*) * NUM_MAX_P2P_PEERS);
            sycl_ctx->comm_queue().wait();
        }

        // Initialize iSHMEM and allocate RDMA buffer
#ifdef SYCL_ISHMEM
        if (num_rdma_bytes > 0) {
            // Bootstrap iSHMEM via PMI
            int pe = ishmem_runtime::init();
            ishmem_initialized = true;

            // Verify PE assignment matches expected rank
            auto ishmem_rank = low_latency_mode ? rank : rdma_rank;
            EP_HOST_ASSERT(pe == ishmem_rank and
                           "iSHMEM PE does not match expected RDMA rank");
            ishmem_runtime::barrier_all();

            // Allocate RDMA buffer from symmetric heap
            rdma_buffer_ptr = ishmem_runtime::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

            // Zero RDMA buffer (important for low-latency mode)
            sycl_ctx->comm_queue().memset(rdma_buffer_ptr, 0, num_rdma_bytes).wait();

            // Allocate shrink mode buffers
            if (enable_shrink) {
                int num_mask_bytes = num_ranks * sizeof(int);
                int num_sync_bytes = num_ranks * sizeof(int);
                mask_buffer_ptr = reinterpret_cast<int*>(
                    ishmem_runtime::alloc(num_mask_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
                sync_buffer_ptr = reinterpret_cast<int*>(
                    ishmem_runtime::alloc(num_sync_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
                sycl_ctx->comm_queue().memset(mask_buffer_ptr, 0, num_mask_bytes);
                sycl_ctx->comm_queue().memset(sync_buffer_ptr, 0, num_sync_bytes);
                sycl_ctx->comm_queue().wait();
            }

            ishmem_runtime::barrier_all();
        }
#endif

        available = true;
    }

    // -----------------------------------------------------------------------
    // destroy — release all resources
    // -----------------------------------------------------------------------
    void destroy() {
        if (destroyed) return;

        // Wait for all pending operations
        if (sycl_ctx) {
            sycl_ctx->comm_queue().wait();
            sycl_ctx->default_queue().wait();
        }

        // Close intranode IPC handles
        if (num_p2p_bytes > 0 && is_available()) {
            for (int i = 0; i < num_p2p_ranks; ++i) {
                if (i != p2p_rank && buffer_ptrs[i] != nullptr) {
                    mem_allocator->close_ipc_handle(buffer_ptrs[i]);
                    buffer_ptrs[i] = nullptr;
                }
            }
        }

        // Free local P2P buffer
        if (num_p2p_bytes > 0 && buffer_ptrs[p2p_rank] != nullptr) {
            mem_allocator->free(buffer_ptrs[p2p_rank]);
            buffer_ptrs[p2p_rank] = nullptr;
        }

        // Free iSHMEM resources
#ifdef SYCL_ISHMEM
        if (ishmem_initialized && is_available() && num_rdma_bytes > 0) {
            ishmem_runtime::barrier_all();
            if (rdma_buffer_ptr) {
                ishmem_runtime::free(rdma_buffer_ptr);
                rdma_buffer_ptr = nullptr;
            }
            if (enable_shrink) {
                if (mask_buffer_ptr) {
                    ishmem_runtime::free(mask_buffer_ptr);
                    mask_buffer_ptr = nullptr;
                }
                if (sync_buffer_ptr) {
                    ishmem_runtime::free(sync_buffer_ptr);
                    sync_buffer_ptr = nullptr;
                }
            }
            ishmem_runtime::finalize();
            ishmem_initialized = false;
        }
#endif

        // Free workspace
        if (workspace) {
            mem_allocator->free(workspace);
            workspace = nullptr;
        }

        // Free host-mapped counters
        if (moe_recv_counter) {
            host_allocator->free_host(const_cast<int*>(moe_recv_counter));
            moe_recv_counter = nullptr;
        }
        if (moe_recv_expert_counter) {
            host_allocator->free_host(const_cast<int*>(moe_recv_expert_counter));
            moe_recv_expert_counter = nullptr;
        }
        if (moe_recv_rdma_counter) {
            host_allocator->free_host(const_cast<int*>(moe_recv_rdma_counter));
            moe_recv_rdma_counter = nullptr;
        }

        destroyed = true;
        available = false;
    }

    // -----------------------------------------------------------------------
    // Layout
    // -----------------------------------------------------------------------
    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
    get_dispatch_layout(const torch::Tensor& topk_idx,
                        int num_experts,
                        std::optional<EventHandle>& previous_event,
                        bool async_op,
                        bool allocate_on_comm_stream) {
        throw std::runtime_error("SYCL backend: get_dispatch_layout not yet implemented (phase-9)");
    }

    // -----------------------------------------------------------------------
    // Intranode dispatch / combine
    // -----------------------------------------------------------------------
    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::vector<int>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               std::optional<EventHandle>>
    intranode_dispatch(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx,
                       const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank,
                       const torch::Tensor& is_token_in_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens,
                       bool cached,
                       int num_experts,
                       std::optional<EventHandle>& previous_event,
                       bool async_op,
                       bool allocate_on_comm_stream) {
        throw std::runtime_error("SYCL backend: intranode_dispatch not yet implemented (phase-8)");
    }

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    intranode_combine(const torch::Tensor& x,
                      const std::optional<torch::Tensor>& x_scales,
                      const torch::Tensor& topk_idx,
                      const torch::Tensor& topk_weights,
                      const torch::Tensor& src_idx,
                      const torch::Tensor& channel_recv_offset,
                      const torch::Tensor& recv_head,
                      int num_topk,
                      int hidden,
                      int num_experts,
                      std::optional<EventHandle>& previous_event,
                      bool async_op,
                      bool allocate_on_comm_stream) {
        throw std::runtime_error("SYCL backend: intranode_combine not yet implemented (phase-8)");
    }

    // -----------------------------------------------------------------------
    // Internode dispatch / combine (high-throughput)
    // -----------------------------------------------------------------------
    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               torch::Tensor,
               std::vector<int>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               std::optional<EventHandle>>
    internode_dispatch(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx,
                       const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank,
                       const torch::Tensor& is_token_in_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_expert,
                       const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                       int cached_num_recv_tokens,
                       bool cached,
                       int num_experts,
                       std::optional<EventHandle>& previous_event,
                       bool async_op,
                       bool allocate_on_comm_stream) {
        throw std::runtime_error("SYCL backend: internode_dispatch not yet implemented (phase-6)");
    }

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    internode_combine(const torch::Tensor& x,
                      const std::optional<torch::Tensor>& x_scales,
                      const torch::Tensor& topk_idx,
                      const torch::Tensor& topk_weights,
                      const torch::Tensor& src_idx,
                      const torch::Tensor& channel_recv_offset,
                      const torch::Tensor& recv_head,
                      const torch::Tensor& rdma_channel_recv_offset,
                      const torch::Tensor& rdma_recv_head,
                      int num_topk,
                      int hidden,
                      int num_experts,
                      std::optional<EventHandle>& previous_event,
                      bool async_op,
                      bool allocate_on_comm_stream) {
        throw std::runtime_error("SYCL backend: internode_combine not yet implemented (phase-6)");
    }

    // -----------------------------------------------------------------------
    // Low-latency dispatch / combine
    // -----------------------------------------------------------------------
    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
        throw std::runtime_error("SYCL backend: clean_low_latency_buffer not yet implemented (phase-7)");
    }

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               torch::Tensor,
               torch::Tensor,
               std::optional<EventHandle>>
    low_latency_dispatch(const torch::Tensor& x,
                         const std::optional<torch::Tensor>& x_scales,
                         const torch::Tensor& topk_idx,
                         const torch::Tensor& topk_weights,
                         int num_max_dispatch_tokens_per_rank,
                         int num_experts,
                         bool use_fp8,
                         bool async_op,
                         std::optional<EventHandle>& previous_event) {
        throw std::runtime_error("SYCL backend: low_latency_dispatch not yet implemented (phase-7)");
    }

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    low_latency_combine(const torch::Tensor& x,
                        const torch::Tensor& topk_idx,
                        const torch::Tensor& topk_weights,
                        int num_max_dispatch_tokens_per_rank,
                        int hidden,
                        int num_experts,
                        bool async_op,
                        std::optional<EventHandle>& previous_event) {
        throw std::runtime_error("SYCL backend: low_latency_combine not yet implemented (phase-7)");
    }

    torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank,
                                                      int hidden,
                                                      int num_experts) const {
        throw std::runtime_error("SYCL backend: get_next_low_latency_combine_buffer not yet implemented (phase-7)");
    }

    void low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
        throw std::runtime_error("SYCL backend: low_latency_update_mask_buffer not yet implemented (phase-7)");
    }

    void low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
        throw std::runtime_error("SYCL backend: low_latency_query_mask_buffer not yet implemented (phase-7)");
    }

    void low_latency_clean_mask_buffer() {
        throw std::runtime_error("SYCL backend: low_latency_clean_mask_buffer not yet implemented (phase-7)");
    }
};

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------
bool is_sm90_compiled() {
    // SM90 is an NVIDIA concept; on SYCL backend always return false
    return false;
}

}  // namespace deep_ep

// ---------------------------------------------------------------------------
// pybind11 module — same interface as CUDA backend
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library (SYCL + iSHMEM backend)";

    py::class_<deep_ep::Config>(m, "Config")
        .def(py::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6,
             py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6,
             py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_p2p_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    py::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(py::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    py::class_<deep_ep::Buffer>(m, "Buffer")
        .def(py::init<int, int, int64_t, int64_t, bool, bool, bool, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_internode_unique_id", &deep_ep::Buffer::get_local_internode_unique_id)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &deep_ep::Buffer::get_comm_stream)
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("low_latency_update_mask_buffer", &deep_ep::Buffer::low_latency_update_mask_buffer)
        .def("low_latency_query_mask_buffer", &deep_ep::Buffer::low_latency_query_mask_buffer)
        .def("low_latency_clean_mask_buffer", &deep_ep::Buffer::low_latency_clean_mask_buffer)
        .def("get_next_low_latency_combine_buffer", &deep_ep::Buffer::get_next_low_latency_combine_buffer);

    m.def("is_sm90_compiled", deep_ep::is_sm90_compiled);

    // topk_idx_t attribute — export the scalar type to Python
    m.attr("topk_idx_t") =
        py::reinterpret_borrow<py::object>((PyObject*)torch::getTHPDtype(
            c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value));

    // Backend info
    m.attr("backend") = "sycl_ishmem";
#ifdef SYCL_ISHMEM
    m.attr("ishmem_available") = true;
#else
    m.attr("ishmem_available") = false;
#endif
}
