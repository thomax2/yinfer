#include "qwen_execution_plan.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "llm_engine/memory/workspace.h"
#define __arm__
#define __aarch64__

namespace llm_engine {
namespace {

size_t aligned(size_t bytes) {
    return (bytes + 63) & ~size_t(63);
}

class RMSNormPlanNode final : public KernelNode {
public:
    RMSNormPlanNode(
        std::string name,
        PlanValueId input,
        PlanValueId output,
        const Tensor* weight,
        int hidden_dim,
        float eps)
        : KernelNode(std::move(name), {input}, {output}),
          input_(input), output_(output), weight_(weight),
          hidden_dim_(hidden_dim), eps_(eps) {}

    Status run(ExecutionContext& context) override {
#if !defined(__aarch64__) && !defined(__arm__)
        (void)context;
        return Status::INVALID_ARGUMENT;
#else
        return arm_neon::rmsnorm_f16_batch_neon(
            context.ptr<fp16_t>(input_),
            weight_->ptr<fp16_t>(),
            context.ptr<fp16_t>(output_),
            context.actual_rows,
            hidden_dim_,
            eps_);
#endif
    }

private:
    PlanValueId input_;
    PlanValueId output_;
    const Tensor* weight_;
    int hidden_dim_;
    float eps_;
};

class AttentionPlanNode final : public KernelNode {
public:
    AttentionPlanNode(
        std::string name,
        PlanValueId input,
        PlanValueId output,
        PlanValueId scratch,
        const QwenBlockWeights* weights,
        int layer_id,
        ExecutionMode mode,
        arm_neon::AttentionConfig config,
        size_t scratch_bytes)
        : KernelNode(
              // 大 scratch 先进入 planner，避免小 output 从可复用大块头部切走后
              // 剩余空间略小于下一层 scratch 而产生第二块大区间。
              std::move(name), {input}, {scratch, output},
              ParallelismPolicy::INTERNAL_THREAD_POOL),
          input_(input), output_(output), scratch_(scratch),
          weights_(weights), layer_id_(layer_id), config_(config),
          mode_(mode), scratch_bytes_(scratch_bytes) {}

    Status run(ExecutionContext& context) override {
#if !defined(__aarch64__) && !defined(__arm__)
        (void)context;
        return Status::INVALID_ARGUMENT;
#else
        auto* runtime = static_cast<QwenPlanRuntime*>(context.user_data);
        if (!runtime || !runtime->kv_cache || !runtime->cos || !runtime->sin) {
            return Status::INVALID_ARGUMENT;
        }
        Tensor input(
            {context.actual_rows, config_.hidden_dim},
            context.ptr<fp16_t>(input_), DataType::FP16);
        Tensor output(
            {context.actual_rows, config_.hidden_dim},
            context.ptr<fp16_t>(output_), DataType::FP16);
        Workspace workspace(context.value_data(scratch_), scratch_bytes_);
        if (mode_ == ExecutionMode::PREFILL) {
            return arm_neon::attention_f16_gptq_prefill_neon(
                input, output,
                weights_->q_proj, weights_->k_proj,
                weights_->v_proj, weights_->o_proj,
                weights_->b_q.ptr<fp16_t>(),
                weights_->b_k.ptr<fp16_t>(),
                weights_->b_v.ptr<fp16_t>(),
                runtime->cos, runtime->sin,
                *runtime->kv_cache, layer_id_, runtime->start_pos,
                config_, workspace);
        }
        return arm_neon::attention_f16_gptq_neon(
            input, output,
            weights_->q_proj, weights_->k_proj,
            weights_->v_proj, weights_->o_proj,
            weights_->b_q.ptr<fp16_t>(),
            weights_->b_k.ptr<fp16_t>(),
            weights_->b_v.ptr<fp16_t>(),
            runtime->cos, runtime->sin,
            *runtime->kv_cache, layer_id_, runtime->current_pos,
            config_, workspace);
#endif
    }

private:
    PlanValueId input_;
    PlanValueId output_;
    PlanValueId scratch_;
    const QwenBlockWeights* weights_;
    int layer_id_;
    arm_neon::AttentionConfig config_;
    ExecutionMode mode_;
    size_t scratch_bytes_;
};

class AddPlanNode final : public KernelNode {
public:
    AddPlanNode(
        std::string name,
        PlanValueId left,
        PlanValueId right,
        PlanValueId output,
        int hidden_dim)
        : KernelNode(std::move(name), {left, right}, {output}),
          left_(left), right_(right), output_(output), hidden_dim_(hidden_dim) {}

    Status run(ExecutionContext& context) override {
#if !defined(__aarch64__) && !defined(__arm__)
        (void)context;
        return Status::INVALID_ARGUMENT;
#else
        arm_neon::add_f16_batch_neon(
            context.ptr<fp16_t>(left_),
            context.ptr<fp16_t>(right_),
            context.ptr<fp16_t>(output_),
            static_cast<size_t>(context.actual_rows) * hidden_dim_);
        return Status::SUCCESS;
#endif
    }

private:
    PlanValueId left_;
    PlanValueId right_;
    PlanValueId output_;
    int hidden_dim_;
};

class FFNPlanNode final : public KernelNode {
public:
    FFNPlanNode(
        std::string name,
        PlanValueId input,
        PlanValueId output,
        PlanValueId scratch,
        const QwenBlockWeights* weights,
        ExecutionMode mode,
        arm_neon::FFNConfig config,
        size_t scratch_bytes)
        : KernelNode(
              std::move(name), {input}, {scratch, output},
              ParallelismPolicy::INTERNAL_THREAD_POOL),
          input_(input), output_(output), scratch_(scratch),
          weights_(weights), mode_(mode), config_(config), scratch_bytes_(scratch_bytes) {}

    Status run(ExecutionContext& context) override {
#if !defined(__aarch64__) && !defined(__arm__)
        (void)context;
        return Status::INVALID_ARGUMENT;
#else
        Tensor input(
            {context.actual_rows, config_.hidden_dim},
            context.ptr<fp16_t>(input_), DataType::FP16);
        Tensor output(
            {context.actual_rows, config_.hidden_dim},
            context.ptr<fp16_t>(output_), DataType::FP16);
        Workspace workspace(context.value_data(scratch_), scratch_bytes_);
        if (mode_ == ExecutionMode::PREFILL) {
            return arm_neon::ffn_f16_gptq_batch_neon(
                input, output,
                weights_->gate_proj, weights_->up_proj, weights_->down_proj,
                config_, workspace);
        }
        return arm_neon::ffn_f16_gptq_neon(
            input, output,
            weights_->gate_proj, weights_->up_proj, weights_->down_proj,
            config_, workspace);
#endif
    }

private:
    PlanValueId input_;
    PlanValueId output_;
    PlanValueId scratch_;
    const QwenBlockWeights* weights_;
    ExecutionMode mode_;
    arm_neon::FFNConfig config_;
    size_t scratch_bytes_;
};

class ArgmaxPlanNode final : public KernelNode {
public:
    ArgmaxPlanNode(
        PlanValueId input,
        PlanValueId output,
        const arm_neon::GPTQInt8Weight* weight)
        : KernelNode(
              "lm_head.fused_argmax", {input}, {output},
              ParallelismPolicy::INTERNAL_THREAD_POOL),
          input_(input), output_(output), weight_(weight) {}

    Status run(ExecutionContext& context) override {
#if !defined(__aarch64__) && !defined(__arm__)
        (void)context;
        return Status::INVALID_ARGUMENT;
#else
        arm_neon::ArgmaxResult result =
            arm_neon::linear_gptq_int8_decode_argmax_neon(
                context.ptr<fp16_t>(input_), *weight_, nullptr, 0);
        *context.ptr<arm_neon::ArgmaxResult>(output_) = result;
        return result.index >= 0 ? Status::SUCCESS : Status::INVALID_ARGUMENT;
#endif
    }

private:
    PlanValueId input_;
    PlanValueId output_;
    const arm_neon::GPTQInt8Weight* weight_;
};

} // namespace

namespace {

QwenPlanArtifacts build_qwen_plan(
    ExecutionMode mode,
    int row_capacity,
    const std::vector<QwenBlockWeights>& layers,
    const Tensor& final_norm_weight,
    const arm_neon::GPTQInt8Weight& lm_head,
    const arm_neon::AttentionConfig& attention_config,
    const arm_neon::FFNConfig& ffn_config,
    int max_seq_len,
    float rms_norm_eps) {
    if (mode != ExecutionMode::SINGLE_DECODE && mode != ExecutionMode::PREFILL) {
        throw std::invalid_argument("unsupported Qwen plan mode");
    }
    if (row_capacity <= 0 || attention_config.hidden_dim <= 0 ||
        ffn_config.intermediate_size <= 0 || max_seq_len <= 0 || layers.empty()) {
        throw std::invalid_argument("invalid Qwen plan configuration");
    }

    ExecutablePlanBuilder builder({mode, row_capacity});
    const size_t hidden_row_bytes =
        static_cast<size_t>(attention_config.hidden_dim) * sizeof(fp16_t);
    const size_t hidden_bytes = static_cast<size_t>(row_capacity) * hidden_row_bytes;
    const int q_size = attention_config.num_q_heads * attention_config.head_dim;
    const int kv_size = attention_config.num_kv_heads * attention_config.head_dim;
    const int num_rep = attention_config.num_q_heads / attention_config.num_kv_heads;

    QwenPlanArtifacts artifacts;
    artifacts.hidden_input = builder.add_value({"hidden.input", hidden_bytes, 64, true});
    PlanValueId current_hidden = artifacts.hidden_input;

    for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
        const QwenBlockWeights* weights = &layers[layer_id];
        const std::string prefix = "layer." + std::to_string(layer_id) + ".";

        PlanValueId norm1 = builder.add_value({prefix + "norm1", hidden_bytes});
        builder.add_node(std::make_unique<RMSNormPlanNode>(
            prefix + "rmsnorm1", current_hidden, norm1,
            &weights->norm1_w, attention_config.hidden_dim, rms_norm_eps));

        const size_t rows = static_cast<size_t>(row_capacity);
        const size_t attention_scratch =
            aligned(rows * q_size * sizeof(fp16_t)) +
            aligned(rows * kv_size * sizeof(fp16_t)) * 2 +
            aligned(static_cast<size_t>(num_rep) * max_seq_len * sizeof(fp16_t)) +
            aligned(rows * q_size * sizeof(fp16_t)) + 5 * 64;
        PlanValueId attn_scratch = builder.add_value(
            {prefix + "attention.scratch", attention_scratch});
        PlanValueId attn_out = builder.add_value({prefix + "attention.out", hidden_bytes});
        builder.add_node(std::make_unique<AttentionPlanNode>(
            prefix + "attention", norm1, attn_out, attn_scratch,
            weights, static_cast<int>(layer_id), mode, attention_config,
            attention_scratch));

        PlanValueId after_attention =
            builder.add_value({prefix + "hidden.after_attention", hidden_bytes});
        builder.add_node(std::make_unique<AddPlanNode>(
            prefix + "residual.attention", current_hidden, attn_out,
            after_attention, attention_config.hidden_dim));

        PlanValueId norm2 = builder.add_value({prefix + "norm2", hidden_bytes});
        builder.add_node(std::make_unique<RMSNormPlanNode>(
            prefix + "rmsnorm2", after_attention, norm2,
            &weights->norm2_w, attention_config.hidden_dim, rms_norm_eps));

        const size_t activation_copies = mode == ExecutionMode::PREFILL ? 2 : 1;
        const size_t ffn_scratch =
            activation_copies * aligned(
                rows * ffn_config.intermediate_size * sizeof(fp16_t)) + 64;
        PlanValueId ffn_scratch_value =
            builder.add_value({prefix + "ffn.scratch", ffn_scratch});
        PlanValueId ffn_out = builder.add_value({prefix + "ffn.out", hidden_bytes});
        builder.add_node(std::make_unique<FFNPlanNode>(
            prefix + (mode == ExecutionMode::PREFILL
                ? "ffn.batch" : "ffn.fused_gate_up_swiglu_down"),
            norm2, ffn_out, ffn_scratch_value,
            weights, mode, ffn_config, ffn_scratch));

        PlanValueId layer_output =
            builder.add_value({prefix + "hidden.output", hidden_bytes});
        builder.add_node(std::make_unique<AddPlanNode>(
            prefix + "residual.ffn", after_attention, ffn_out,
            layer_output, attention_config.hidden_dim));
        current_hidden = layer_output;
    }

    // Prefill 只对最后一个 token 做 final norm 和 LM Head；Decode 的最后一行就是唯一一行。
    artifacts.norm_output = builder.add_value(
        {"final_norm.output", hidden_row_bytes, 64, true});
    class FinalRowRMSNormNode final : public KernelNode {
    public:
        FinalRowRMSNormNode(
            PlanValueId input, PlanValueId output, const Tensor* weight,
            int hidden_dim, float eps)
            : KernelNode("final_norm.last_row", {input}, {output}),
              input_(input), output_(output), weight_(weight),
              hidden_dim_(hidden_dim), eps_(eps) {}
        Status run(ExecutionContext& context) override {
#if !defined(__aarch64__) && !defined(__arm__)
            (void)context;
            return Status::INVALID_ARGUMENT;
#else
            const fp16_t* input = context.ptr<fp16_t>(input_) +
                static_cast<size_t>(context.actual_rows - 1) * hidden_dim_;
            arm_neon::rmsnorm_f16_neon(
                input, weight_->ptr<fp16_t>(), context.ptr<fp16_t>(output_),
                hidden_dim_, eps_);
            return Status::SUCCESS;
#endif
        }
    private:
        PlanValueId input_;
        PlanValueId output_;
        const Tensor* weight_;
        int hidden_dim_;
        float eps_;
    };
    builder.add_node(std::make_unique<FinalRowRMSNormNode>(
        current_hidden, artifacts.norm_output,
        &final_norm_weight, attention_config.hidden_dim, rms_norm_eps));

    artifacts.argmax_output = builder.add_value(
        {"lm_head.argmax", sizeof(arm_neon::ArgmaxResult), 64, true});
    builder.add_node(std::make_unique<ArgmaxPlanNode>(
        artifacts.norm_output, artifacts.argmax_output, &lm_head));

    artifacts.plan = std::make_unique<ExecutablePlan>(builder.compile());
    const size_t row_bytes = aligned(hidden_row_bytes);
    artifacts.coarse_workspace_bytes = 2 * row_bytes + 64ULL * 1024 * 1024 + 64;
    return artifacts;
}

} // namespace

QwenPlanArtifacts build_qwen_single_decode_plan(
    const std::vector<QwenBlockWeights>& layers,
    const Tensor& final_norm_weight,
    const arm_neon::GPTQInt8Weight& lm_head,
    const arm_neon::AttentionConfig& attention_config,
    const arm_neon::FFNConfig& ffn_config,
    int max_seq_len,
    float rms_norm_eps) {
    return build_qwen_plan(
        ExecutionMode::SINGLE_DECODE, 1,
        layers, final_norm_weight, lm_head,
        attention_config, ffn_config, max_seq_len, rms_norm_eps);
}

QwenPlanArtifacts build_qwen_prefill_plan(
    const std::vector<QwenBlockWeights>& layers,
    const Tensor& final_norm_weight,
    const arm_neon::GPTQInt8Weight& lm_head,
    const arm_neon::AttentionConfig& attention_config,
    const arm_neon::FFNConfig& ffn_config,
    int max_seq_len,
    int row_capacity,
    float rms_norm_eps) {
    return build_qwen_plan(
        ExecutionMode::PREFILL, row_capacity,
        layers, final_norm_weight, lm_head,
        attention_config, ffn_config, max_seq_len, rms_norm_eps);
}

} // namespace llm_engine
