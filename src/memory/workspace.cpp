#include "llm_engine/memory/workspace.h"
#include "llm_engine/memory/memory_pool.h"

namespace llm_engine {

Workspace::Workspace(size_t bytes)
{
    ptr = g_memory_pool->allocate(bytes);
    capacity = bytes;
}

void* Workspace::data()
{
    return ptr;
}

size_t Workspace::size()
{
    return capacity;
}

}