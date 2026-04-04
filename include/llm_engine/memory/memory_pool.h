#pragma once

#include <cstddef>

namespace llm_engine {

struct alignas(16) BlockHeader {
    size_t size;        // 数据块大小（不含header）
    bool free;          // 是否空闲
    
    BlockHeader* next;  // 下一个块
    BlockHeader* prev;  // 上一个块
};

class MemoryPool {

public:

    MemoryPool(size_t bytes);
    ~MemoryPool();

    void* allocate(size_t bytes);
    void free_block(void* ptr);

private:

    void* arena;
    size_t arena_size;

    BlockHeader* head;

    BlockHeader* find_best_fit(size_t bytes);
    void split(BlockHeader* block, size_t bytes);
    void coalesce(BlockHeader* block);
};

extern MemoryPool* g_memory_pool;

}