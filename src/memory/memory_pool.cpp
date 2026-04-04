#include "llm_engine/memory/memory_pool.h"

#include <cstdlib>

namespace llm_engine {

MemoryPool* g_memory_pool = nullptr;     // 需要被初始化为一个 MemoryPool 实例，通常在程序启动时进行初始化。

/*
构造函数：初始化内存池，分配一个大的内存块作为 arena，并设置初始的 BlockHeader。
双向链表
*/
MemoryPool::MemoryPool(size_t bytes)
{
    arena = std::malloc(bytes);
    arena_size = bytes;

    head = (BlockHeader*)arena;
    head->size = bytes - sizeof(BlockHeader);
    head->free = true;
    head->next = nullptr;
    head->prev = nullptr;
}

MemoryPool::~MemoryPool()
{
    std::free(arena);
}

BlockHeader* MemoryPool::find_best_fit(size_t bytes) 
{
    BlockHeader* best = nullptr;

    for(BlockHeader* cur = head; cur; cur = cur->next) {
        if(cur->free && cur->size >= bytes) {
            if(!best || cur->size < best->size) {
                best = cur;
            }
        }
    }
    
    return best;
}

void MemoryPool::split(BlockHeader* block, size_t bytes)
{
    if(block->size <= bytes + sizeof(BlockHeader))
        return; // 不够分割

    char *base = (char*)block;
    BlockHeader* new_block = (BlockHeader*)(base + sizeof(BlockHeader) + bytes);

    
    new_block->size = block->size - bytes - sizeof(BlockHeader);
    new_block->free = true;
    new_block->next = block->next;
    new_block->prev = block;

    block->size = bytes;
    if(block->next)
        block->next->prev = new_block;
    block->next = new_block;
}

void* MemoryPool::allocate(size_t bytes) {
    size_t aligned_bytes = (bytes + 15) & ~15;
    BlockHeader* block = find_best_fit(aligned_bytes);

    if(!block)
        return nullptr; // 没有足够的内存
    
    split(block, aligned_bytes);
    block->free = false;

    return (char*)block + sizeof(BlockHeader);
}

void MemoryPool::free_block(void* ptr) {
    if(!ptr)
        return;

    BlockHeader* block = (BlockHeader*)((char*)ptr - sizeof(BlockHeader));
    block->free = true;
    coalesce(block);
}

// 合并块
void MemoryPool::coalesce(BlockHeader* block) {
    if(block->next && block->next->free) {
        block->size += sizeof(BlockHeader) + block->next->size;
        block->next = block->next->next;
        if(block->next)
            block->next->prev = block;
    }
    if(block->prev && block->prev->free) {
        coalesce(block->prev);
    }
}

}