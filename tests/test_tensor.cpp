#include <gtest/gtest.h>
#include "llm_engine/tensor.h"

using namespace llm_engine;

TEST(TensorTest, StrideCompute) {

    Tensor t({2,3,4});

    EXPECT_EQ(t.stride[0], 12);
    EXPECT_EQ(t.stride[1], 4);
    EXPECT_EQ(t.stride[2], 1);

}

TEST(TensorTest, MemoryAllocate) {

    Tensor t({2,2});

    ASSERT_NE(t.data, nullptr);

    float* ptr = t.ptr<float>();

    ptr[0] = 1.0f;
    ptr[1] = 2.0f;

    EXPECT_EQ(ptr[0], 1.0f);
    EXPECT_EQ(ptr[1], 2.0f);

}