#pragma once

// Level Zero IPC memory allocator for DeepEP SYCL backend.
// Replaces CUDA's cudaIpcMemHandle / CUmemFabricHandle with L0 IPC handles.

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "sycl_backend/exception.hpp"
#include "sycl_backend/sycl_context.hpp"

namespace deep_ep {

// ---------------------------------------------------------------------------
// L0IpcMemHandle — serializable IPC handle (analogous to shared_memory::MemHandle)
// ---------------------------------------------------------------------------
struct L0IpcMemHandle {
    ze_ipc_mem_handle_t ipc_handle;   // 64 bytes opaque
    size_t size;                       // allocation size
};

static constexpr size_t L0_IPC_HANDLE_SIZE = sizeof(L0IpcMemHandle);

// ---------------------------------------------------------------------------
// L0MemoryAllocator — device memory allocation with IPC sharing
// ---------------------------------------------------------------------------
class L0MemoryAllocator {
public:
    explicit L0MemoryAllocator(SyclContext& ctx) : ctx_(ctx) {}

    // Allocate device memory via SYCL USM (device)
    void malloc(void** ptr, size_t size) {
        *ptr = sycl::malloc_device(size, ctx_.device(), ctx_.context());
        EP_HOST_ASSERT(*ptr != nullptr and "sycl::malloc_device failed");
    }

    void free(void* ptr) {
        if (ptr) {
            sycl::free(ptr, ctx_.context());
        }
    }

    // Zero-fill memory asynchronously on the communication queue
    void memset_async(void* ptr, int value, size_t size, sycl::queue& queue) {
        queue.memset(ptr, value, size);
    }

    // Get IPC handle for a device allocation
    void get_ipc_handle(L0IpcMemHandle* handle, void* ptr, size_t size) {
        handle->size = size;
        ZE_CHECK(zeMemGetIpcHandle(ctx_.ze_context(), ptr, &handle->ipc_handle));
    }

    // Open a remote IPC handle
    void open_ipc_handle(void** ptr, const L0IpcMemHandle* handle) {
        ZE_CHECK(zeMemOpenIpcHandle(
            ctx_.ze_context(),
            ctx_.ze_device(),
            handle->ipc_handle,
            0,  // flags
            ptr));
    }

    // Close a remote IPC handle
    void close_ipc_handle(void* ptr) {
        ZE_CHECK(zeMemCloseIpcHandle(ctx_.ze_context(), ptr));
    }

private:
    SyclContext& ctx_;
};

// ---------------------------------------------------------------------------
// Host-pinned memory (USM host allocations)
// ---------------------------------------------------------------------------
class L0HostAllocator {
public:
    explicit L0HostAllocator(SyclContext& ctx) : ctx_(ctx) {}

    // Allocate pinned host memory accessible from device
    void* malloc_host(size_t size) {
        void* ptr = sycl::malloc_host(size, ctx_.context());
        EP_HOST_ASSERT(ptr != nullptr and "sycl::malloc_host failed");
        return ptr;
    }

    void free_host(void* ptr) {
        if (ptr) {
            sycl::free(ptr, ctx_.context());
        }
    }

private:
    SyclContext& ctx_;
};

}  // namespace deep_ep
