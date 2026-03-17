#include <gtest/gtest.h>

#include "llm_engine/graph/graph.h"
#include "llm_engine/graph/compiler.h"
#include "llm_engine/memory/memory_pool.h"

using namespace llm_engine;

static void init_memory_pool() {
    static bool initialized = false;
    if (!initialized) {
        g_memory_pool = new MemoryPool(512ULL * 1024 * 1024);
        initialized = true;
    }
}

TEST(GraphTest, MatmulAddChain)
{
    init_memory_pool();

    ComputationGraph g;

    auto* A = g.create_tensor({2,2});
    auto* B = g.create_tensor({2,2});
    auto* C = g.create_tensor({2,2});

    auto* Tmp = g.create_tensor({2,2});
    auto* D = g.create_tensor({2,2});

    float* a = A->ptr<float>();
    float* b = B->ptr<float>();
    float* c = C->ptr<float>();

    a[0]=1; a[1]=2;
    a[2]=3; a[3]=4;

    b[0]=5; b[1]=6;
    b[2]=7; b[3]=8;

    c[0]=1; c[1]=1;
    c[2]=1; c[3]=1;

    g.add_add(Tmp, C, D);
    g.add_matmul(A, B, Tmp);

    GraphCompiler compiler;
    auto plan = compiler.compile(g);

    GraphRuntime runtime;
    CHECK_STATUS(runtime.run(plan));

    float* d = D->ptr<float>();

    EXPECT_EQ(d[0],20);
    EXPECT_EQ(d[1],23);
    EXPECT_EQ(d[2],44);
    EXPECT_EQ(d[3],51);
}