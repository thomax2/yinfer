#include <gtest/gtest.h>
#include "llm_engine/tensor.h"
#include "backends/cpu/reference/math_ref.h"

using namespace llm_engine;
using namespace llm_engine::reference;

static void init_memory_pool() {
    static bool initialized = false;
    if (!initialized) {
        g_memory_pool = new MemoryPool(512ULL * 1024 * 1024);
        initialized = true;
    }
}

TEST(MatmulTest, Basic2x2) {
    init_memory_pool();
    Tensor A({2,2}, DataType::FP32);
    Tensor B({2,2}, DataType::FP32);
    Tensor C({2,2}, DataType::FP32);
    A.ensure_allocated();
    B.ensure_allocated();
    C.ensure_allocated();

    float* a = A.ptr<float>();
    float* b = B.ptr<float>();

    a[0]=1; a[1]=2;
    a[2]=3; a[3]=4;

    b[0]=5; b[1]=6;
    b[2]=7; b[3]=8;

    CHECK_STATUS(matmul_ref(A,B,C));

    float* c = C.ptr<float>();

    EXPECT_EQ(c[0],19);
    EXPECT_EQ(c[1],22);
    EXPECT_EQ(c[2],43);
    EXPECT_EQ(c[3],50);

}


TEST(AddTest, BasicAdd) {
    init_memory_pool();     
    Tensor A({4}, DataType::FP32);
    Tensor B({4}, DataType::FP32);
    Tensor C({4}, DataType::FP32);
    A.ensure_allocated();
    B.ensure_allocated();
    C.ensure_allocated();

    float* a=A.ptr<float>();
    float* b=B.ptr<float>();

    for(int i=0;i<4;i++){
        a[i]=i;
        b[i]=i;
    }

    CHECK_STATUS(add_ref(A,B,C));

    float* c=C.ptr<float>();

    EXPECT_EQ(c[0],0);
    EXPECT_EQ(c[1],2);
    EXPECT_EQ(c[2],4);
    EXPECT_EQ(c[3],6);

}
