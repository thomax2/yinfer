#pragma once

#include <cstddef>
#include <cstdint>

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

    // 1. 原有的构造函数：从全局内存池真正分配内存
    Workspace(size_t bytes);

    // 2. ⭐新增构造函数：用现有的指针划分子空间 (不分配新内存)
    Workspace(void* pre_allocated_ptr, size_t bytes);
    ~Workspace();

    void* data();
    size_t size();

private:

    void* ptr;
    size_t capacity;
    bool owns_memory; // 标记是否拥有内存的所有权
};

}