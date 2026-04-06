#include "llm_engine/memory/workspace.h"
#include "llm_engine/memory/memory_pool.h"

namespace llm_engine {

Workspace::Workspace(size_t bytes)
{
    ptr = g_memory_pool->allocate(bytes);
    capacity = bytes;
    owns_memory = true;
}

void* Workspace::data()
{
    return ptr;
}

// ⭐新增逻辑：直接使用传入的指针 (充当 Sub-Workspace)
Workspace::Workspace(void* pre_allocated_ptr, size_t bytes)
{
    ptr = pre_allocated_ptr;
    capacity = bytes;
    owns_memory = false; // 只是借用，不拥有所有权
}

Workspace::~Workspace() {
    // 只有真正拥有这块内存的 Workspace（也就是第一手申请的），才负责把它还给内存池
    if (owns_memory && ptr != nullptr) {
        g_memory_pool->free_block(ptr);
        ptr = nullptr;
    }
}

size_t Workspace::size()
{
    return capacity;
}

}