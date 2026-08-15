#include <gtest/gtest.h>
#include "llm_engine/tensor.h"
#include "llm_engine/memory/memory_pool.h"

using namespace llm_engine;

static void init_memory_pool() {
    static bool initialized = false;
    if (!initialized) {
        g_memory_pool = new MemoryPool(512ULL * 1024 * 1024);
        initialized = true;
    }
}

TEST(TensorTest, StrideCompute) {
    init_memory_pool();
    Tensor t({2,3,4});

    EXPECT_EQ(t.stride[0], 12);
    EXPECT_EQ(t.stride[1], 4);
    EXPECT_EQ(t.stride[2], 1);

}

TEST(TensorTest, MemoryAllocate) {
    init_memory_pool();
    Tensor t({2,2}, DataType::FP32);
    EXPECT_EQ(t.data, nullptr);

    t.ensure_allocated();

    ASSERT_NE(t.data, nullptr);
    EXPECT_TRUE(t.owns_data);

    float* ptr = t.ptr<float>();

    ptr[0] = 1.0f;
    ptr[1] = 2.0f;

    EXPECT_EQ(ptr[0], 1.0f);
    EXPECT_EQ(ptr[1], 2.0f);

}
