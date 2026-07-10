#include <gtest/gtest.h>

#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/kernel_common.h"
#include "llm_engine/runtime/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using llm_engine::DataType;
using llm_engine::Status;
using llm_engine::fp16_t;
using llm_engine::arm_neon::ArgmaxResult;
using llm_engine::arm_neon::GPTQInt8Weight;

struct TestWeight {
    GPTQInt8Weight weight;
    std::vector<int8_t> q;
    std::vector<fp16_t> scale;
    std::vector<int8_t> zero;

    TestWeight(int K, int N) {
        weight.K = K;
        weight.N = N;
        weight.group_size = 128;
        weight.num_groups = (K + 127) / 128;
        weight.has_zero = true;
        weight.has_g_idx = false;
        int np = (N + llm_engine::arm_neon::NR_F16 - 1) / llm_engine::arm_neon::NR_F16;
        int k_pad = llm_engine::arm_neon::align_up_int(K, 8);
        q.resize((size_t)np * k_pad * llm_engine::arm_neon::NR_F16);
        scale.resize((size_t)np * weight.num_groups * llm_engine::arm_neon::NR_F16);
        zero.resize((size_t)np * weight.num_groups * llm_engine::arm_neon::NR_F16);
        for (size_t i = 0; i < q.size(); ++i) q[i] = (int8_t)((i * 13 + 5) % 31 - 15);
        for (size_t i = 0; i < scale.size(); ++i) scale[i] = (fp16_t)(0.006f + (i % 7) * 0.001f);
        for (size_t i = 0; i < zero.size(); ++i) zero[i] = (int8_t)((i % 5) - 2);
        weight.qweight_pack.data = q.data();
        weight.qweight_pack.dtype = DataType::INT8;
        weight.qweight_pack.owns_data = false;
        weight.scales_pack.data = scale.data();
        weight.scales_pack.dtype = DataType::FP16;
        weight.scales_pack.owns_data = false;
        weight.zeros_pack.data = zero.data();
        weight.zeros_pack.dtype = DataType::INT8;
        weight.zeros_pack.owns_data = false;
    }
};

class GPTQBatchDecodeTest : public ::testing::Test {
protected:
    llm_engine::ThreadPool pool{4};

    void SetUp() override { llm_engine::g_thread_pool = &pool; }
    void TearDown() override { llm_engine::g_thread_pool = nullptr; }
};

TEST_F(GPTQBatchDecodeTest, LinearMatchesRowReferenceForB2ThroughB8) {
    constexpr int K = 256;
    constexpr int N = 53;
    TestWeight owned(K, N);
    for (int B = 2; B <= 8; ++B) {
        std::vector<fp16_t> x((size_t)B * K);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = (fp16_t)(0.2f * std::sin((float)i * 0.031f));
        }
        std::vector<fp16_t> reference((size_t)B * N);
        std::vector<fp16_t> batch((size_t)B * N);
        for (int row = 0; row < B; ++row) {
            ASSERT_EQ(Status::SUCCESS, llm_engine::arm_neon::linear_gptq_int8_decode_neon(
                x.data() + (size_t)row * K, owned.weight,
                reference.data() + (size_t)row * N, nullptr, nullptr, 0));
        }
        ASSERT_EQ(Status::SUCCESS, llm_engine::arm_neon::linear_gptq_int8_batch_neon(
            x.data(), B, owned.weight, batch.data(), nullptr, nullptr, 0));
        float max_abs = 0.0f;
        for (size_t i = 0; i < batch.size(); ++i) {
            max_abs = std::max(max_abs,
                std::fabs((float)batch[i] - (float)reference[i]));
        }
        EXPECT_LE(max_abs, 0.02f) << "B=" << B;
    }
}

TEST_F(GPTQBatchDecodeTest, BatchedArgmaxMatchesAndDoesNotWriteFullLogits) {
    constexpr int K = 256;
    constexpr int N = 211;
    TestWeight owned(K, N);
    auto before = llm_engine::arm_neon::snapshot_gptq_batch_kernel_stats();
    for (int B = 2; B <= 8; ++B) {
        std::vector<fp16_t> x((size_t)B * K);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = (fp16_t)(0.15f * std::cos((float)i * 0.017f));
        }
        std::vector<ArgmaxResult> reference((size_t)B);
        std::vector<ArgmaxResult> batch((size_t)B);
        for (int row = 0; row < B; ++row) {
            reference[(size_t)row] = llm_engine::arm_neon::linear_gptq_int8_decode_argmax_neon(
                x.data() + (size_t)row * K, owned.weight, nullptr, 0);
        }
        size_t bytes = llm_engine::arm_neon::linear_gptq_int8_batch_argmax_workspace_bytes(
            B, owned.weight);
        std::vector<uint8_t> workspace(bytes);
        ASSERT_EQ(Status::SUCCESS,
            llm_engine::arm_neon::linear_gptq_int8_decode_argmax_batch_neon(
                x.data(), B, owned.weight, batch.data(), workspace.data(), workspace.size()));
        for (int row = 0; row < B; ++row) {
            EXPECT_EQ(reference[(size_t)row].index, batch[(size_t)row].index) << "B=" << B;
        }
    }
    auto delta = llm_engine::arm_neon::diff_gptq_batch_kernel_stats(
        before, llm_engine::arm_neon::snapshot_gptq_batch_kernel_stats());
    EXPECT_GT(delta.argmax_calls, 0u);
    EXPECT_EQ(delta.row_gemv_fallbacks, 0u);
    EXPECT_EQ(delta.full_logits_elements_written, 0u);
}

} // namespace
