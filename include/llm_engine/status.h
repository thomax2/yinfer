#pragma once
#include <iostream>

namespace llm_engine {

enum class Status {
    SUCCESS = 0,
    SHAPE_MISMATCH,
    OUT_OF_MEMORY,
    UNSUPPORTED_DEVICE,
    INVALID_ARGUMENT
};

inline const char* StatusToString(Status s) {
    switch (s) {
        case Status::SUCCESS: return "SUCCESS";
        case Status::SHAPE_MISMATCH: return "SHAPE_MISMATCH";
        case Status::OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case Status::UNSUPPORTED_DEVICE: return "UNSUPPORTED_DEVICE";
        case Status::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        default: return "UNKNOWN";
    }
}

} // namespace llm_engine


#define CHECK_STATUS(expr)                                      \
    do {                                                         \
        llm_engine::Status status = (expr);                      \
        if (status != llm_engine::Status::SUCCESS) {             \
            std::cerr << "Error: "                               \
                      << llm_engine::StatusToString(status)      \
                      << " at " << __FILE__ << ":" << __LINE__   \
                      << std::endl;                              \
            std::abort();                                        \
        }                                                        \
    } while (0)
