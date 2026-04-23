// DeepEP SYCL + iSHMEM backend entry point
// This file provides the same pybind11 module interface as deep_ep.cpp (CUDA backend),
// but targets Intel GPUs via SYCL and iSHMEM for internode communication.
//
// Phase 2: Build system validation — stub implementations.
// Subsequent phases will fill in actual SYCL kernel launches and iSHMEM runtime calls.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <torch/types.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// iSHMEM header — validates that the iSHMEM installation is usable at compile time
#ifdef SYCL_ISHMEM
#include <ishmem.h>
#endif

#include "sycl_backend/configs.hpp"
#include "sycl_backend/exception.hpp"

namespace py = pybind11;

namespace deep_ep {

// ---------------------------------------------------------------------------
// Config — mirrors the CUDA Config struct
// ---------------------------------------------------------------------------
struct Config {
    int num_sms;
    int num_max_nvl_chunked_send_tokens;
    int num_max_nvl_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;

    Config(int num_sms = 20,
           int num_max_nvl_chunked_send_tokens = 6,
           int num_max_nvl_chunked_recv_tokens = 256,
           int num_max_rdma_chunked_send_tokens = 6,
           int num_max_rdma_chunked_recv_tokens = 256)
        : num_sms(num_sms),
          num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
          num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
          num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
          num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {}

    int64_t get_nvl_buffer_size_hint(int num_tokens, int hidden, int num_experts, int num_ranks,
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
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

private:
    int rank, rdma_rank, nvl_rank;
    int num_ranks, num_rdma_ranks, num_nvl_ranks;
    int64_t num_nvl_bytes;
    int64_t num_rdma_bytes;
    bool low_latency_mode;
    bool explicitly_destroy;
    bool enable_shrink;
    bool available = false;
    bool destroyed = false;

public:
    Buffer(int rank,
           int num_ranks,
           int64_t num_nvl_bytes,
           int64_t num_rdma_bytes,
           bool low_latency_mode,
           bool explicitly_destroy,
           bool enable_shrink,
           bool use_fabric)
        : rank(rank),
          num_ranks(num_ranks),
          num_nvl_bytes(num_nvl_bytes),
          num_rdma_bytes(num_rdma_bytes),
          low_latency_mode(low_latency_mode),
          explicitly_destroy(explicitly_destroy),
          enable_shrink(enable_shrink) {
        rdma_rank = rank / NUM_MAX_NVL_PEERS;
        nvl_rank = rank % NUM_MAX_NVL_PEERS;
        num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);
        num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

        // TODO(phase-3): allocate SYCL device memory, iSHMEM symmetric heap, IPC handles
    }

    ~Buffer() noexcept(false) {
        if (not explicitly_destroy) {
            destroy();
        }
    }

    bool is_available() const { return available; }

    int get_num_rdma_ranks() const { return num_rdma_ranks; }

    int get_rdma_rank() const { return rdma_rank; }

    int get_root_rdma_rank(bool global) const { return global ? nvl_rank : 0; }

    int get_local_device_id() const {
        // TODO(phase-3): return actual SYCL device ordinal
        return 0;
    }

    py::bytearray get_local_ipc_handle() const {
        // TODO(phase-3): return Level Zero IPC handle
        std::vector<uint8_t> dummy(128, 0);
        return py::bytearray(reinterpret_cast<const char*>(dummy.data()), dummy.size());
    }

    py::bytearray get_local_internode_unique_id() const {
        // TODO(phase-3): return iSHMEM unique id
        std::vector<uint8_t> dummy(128, 0);
        return py::bytearray(reinterpret_cast<const char*>(dummy.data()), dummy.size());
    }

    py::bytearray get_local_nvshmem_unique_id() const {
        return get_local_internode_unique_id();
    }

    torch::Tensor get_local_buffer_tensor(const py::object& dtype, int64_t offset, bool use_rdma_buffer) const {
        throw std::runtime_error("SYCL backend: get_local_buffer_tensor not yet implemented (phase-3)");
    }

    torch::Stream get_comm_stream() const {
        throw std::runtime_error("SYCL backend: get_comm_stream not yet implemented (phase-3)");
    }

    void sync(const std::vector<int>& device_ids,
              const std::vector<std::optional<py::bytearray>>& all_gathered_handles,
              const std::optional<py::bytearray>& root_unique_id_opt) {
        // TODO(phase-3): open IPC handles, init iSHMEM
        available = true;
    }

    void destroy() {
        if (destroyed) return;
        destroyed = true;
        // TODO(phase-3): free SYCL/iSHMEM resources
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
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
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
