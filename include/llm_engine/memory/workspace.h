#pragma once

#include <cstddef>

namespace llm_engine {

    // 内存对齐辅助函数 (默认 16 字节对齐，完美适配 NEON 128-bit)
    inline char* align_ptr(char* ptr, size_t alignment = 16) {
        return (char*)(((uintptr_t)ptr + (alignment - 1)) & ~(alignment - 1));
    }

    // 辅助计算对齐后的总字节数 (用于提前检查 Workspace 够不够用)
    inline size_t align_size(size_t size, size_t alignment = 16) {
        return (size + (alignment - 1)) & ~(alignment - 1);
    }

class Workspace {

public:

    Workspace(size_t bytes);

    void* data();
    size_t size();

private:

    void* ptr;
    size_t capacity;
};

}