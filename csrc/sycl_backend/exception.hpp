#pragma once

// SYCL-compatible version of kernels/exception.cuh
// No CUDA runtime dependencies.

#include <exception>
#include <string>

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class EPException : public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char* name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char* what() const noexcept override { return message.c_str(); }
};

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                           \
    do {                                                               \
        if (not(cond)) {                                               \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                              \
    } while (0)
#endif

// Level Zero error check
#ifndef ZE_CHECK
#define ZE_CHECK(cmd)                                                                                \
    do {                                                                                             \
        auto _ze_res = (cmd);                                                                        \
        if (_ze_res != 0) {                                                                          \
            throw EPException("ZE", __FILE__, __LINE__, "ze error " + std::to_string(_ze_res));      \
        }                                                                                            \
    } while (0)
#endif
