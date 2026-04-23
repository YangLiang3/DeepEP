#pragma once

// SYCL device and queue management for DeepEP Intel GPU backend.
// Provides a singleton-like context that lazily selects a GPU device
// and maintains a dedicated communication queue.

#include <sycl/sycl.hpp>

#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "sycl_backend/exception.hpp"

namespace deep_ep {

// ---------------------------------------------------------------------------
// SyclContext — per-device SYCL context manager
// ---------------------------------------------------------------------------
class SyclContext {
public:
    // Get or create the SYCL context for a given device ordinal.
    // device_ordinal == -1 means "use the default device" (first GPU).
    static SyclContext& get(int device_ordinal = -1) {
        static std::mutex init_mu;
        static std::vector<SyclContext*> instances;

        std::lock_guard<std::mutex> lock(init_mu);
        int ord = (device_ordinal >= 0) ? device_ordinal : default_device_ordinal();
        if (static_cast<int>(instances.size()) <= ord) {
            instances.resize(ord + 1, nullptr);
        }
        if (instances[ord] == nullptr) {
            instances[ord] = new SyclContext(ord);
        }
        return *instances[ord];
    }

    // Accessors
    sycl::device& device() { return device_; }
    sycl::context& context() { return context_; }
    sycl::queue& default_queue() { return default_queue_; }
    sycl::queue& comm_queue() { return comm_queue_; }
    int ordinal() const { return ordinal_; }
    int num_eus() const { return num_eus_; }

    // Enumerate all Intel GPU devices visible in the platform
    static std::vector<sycl::device> enumerate_gpu_devices() {
        std::vector<sycl::device> gpus;
        for (auto& platform : sycl::platform::get_platforms()) {
            for (auto& dev : platform.get_devices(sycl::info::device_type::gpu)) {
                if (dev.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos) {
                    gpus.push_back(dev);
                }
            }
        }
        return gpus;
    }

    // Get the L0 native handles for IPC operations
    ze_context_handle_t ze_context() const {
        return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context_);
    }

    ze_device_handle_t ze_device() const {
        return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device_);
    }

private:
    sycl::device device_;
    sycl::context context_;
    sycl::queue default_queue_;
    sycl::queue comm_queue_;  // dedicated queue for communication (like CUDA comm_stream)
    int ordinal_;
    int num_eus_;

    explicit SyclContext(int ordinal)
        : ordinal_(ordinal) {
        auto gpus = enumerate_gpu_devices();
        if (gpus.empty()) {
            throw std::runtime_error("No Intel GPU devices found");
        }
        if (ordinal >= static_cast<int>(gpus.size())) {
            throw std::runtime_error(
                "Device ordinal " + std::to_string(ordinal) +
                " out of range (found " + std::to_string(gpus.size()) + " Intel GPUs)");
        }

        device_ = gpus[ordinal];
        context_ = sycl::context(device_);

        // Create in-order queues (matching CUDA stream semantics)
        sycl::property_list props{sycl::property::queue::in_order()};
        default_queue_ = sycl::queue(context_, device_, props);
        comm_queue_ = sycl::queue(context_, device_, props);

        // Query device capabilities
        num_eus_ = device_.get_info<sycl::info::device::max_compute_units>();

        auto name = device_.get_info<sycl::info::device::name>();
        printf("DeepEP SYCL backend: device %d = %s (%d EUs)\n",
               ordinal_, name.c_str(), num_eus_);
    }

    static int default_device_ordinal() {
        const char* env = std::getenv("DEEPEP_DEVICE_ORDINAL");
        if (env) return std::atoi(env);
        return 0;
    }

    SyclContext(const SyclContext&) = delete;
    SyclContext& operator=(const SyclContext&) = delete;
};

}  // namespace deep_ep
