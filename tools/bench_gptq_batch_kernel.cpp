#include "backends/cpu/arm_neon/neon_ops.h"
#include "backends/cpu/arm_neon/kernel_common.h"
#include "llm_engine/runtime/thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using llm_engine::Status;
using llm_engine::fp16_t;
using llm_engine::arm_neon::ArgmaxResult;
using llm_engine::arm_neon::GPTQBatchKernelStats;
using llm_engine::arm_neon::GPTQInt8Weight;

struct OpSpec {
    const char* name;
    const char* prefix;
    int K;
    int N;
    bool argmax;
};

struct OwnedWeight {
    GPTQInt8Weight weight;
    std::vector<int8_t> qweight;
    std::vector<fp16_t> scales;
    std::vector<int8_t> zeros;
};

bool read_exact(const std::string& path, void* data, size_t bytes) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamsize>(bytes)) return false;
    input.seekg(0, std::ios::beg);
    return static_cast<bool>(input.read(static_cast<char*>(data), bytes));
}

std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    return dir.back() == '/' ? dir + file : dir + "/" + file;
}

bool load_weight(const std::string& dir, const OpSpec& spec, OwnedWeight* owned) {
    constexpr int group_size = 128;
    GPTQInt8Weight& w = owned->weight;
    w.K = spec.K;
    w.N = spec.N;
    w.group_size = group_size;
    w.num_groups = (w.K + group_size - 1) / group_size;
    w.has_zero = true;
    w.has_g_idx = false;
    int np = (w.N + llm_engine::arm_neon::NR_F16 - 1) / llm_engine::arm_neon::NR_F16;
    int k_pad = llm_engine::arm_neon::align_up_int(w.K, 8);
    owned->qweight.resize((size_t)np * k_pad * llm_engine::arm_neon::NR_F16);
    owned->scales.resize((size_t)np * w.num_groups * llm_engine::arm_neon::NR_F16);
    owned->zeros.resize((size_t)np * w.num_groups * llm_engine::arm_neon::NR_F16);
    std::string prefix = join_path(dir, spec.prefix);
    if (!read_exact(prefix + ".qweight.s8pack.bin", owned->qweight.data(), owned->qweight.size()) ||
        !read_exact(prefix + ".scales.f16pack.bin", owned->scales.data(),
                    owned->scales.size() * sizeof(fp16_t)) ||
        !read_exact(prefix + ".qzeros.s8pack.bin", owned->zeros.data(), owned->zeros.size())) {
        std::cerr << "failed to load packed weight prefix: " << prefix << std::endl;
        return false;
    }
    w.qweight_pack.data = owned->qweight.data();
    w.qweight_pack.dtype = llm_engine::DataType::INT8;
    w.qweight_pack.owns_data = false;
    w.scales_pack.data = owned->scales.data();
    w.scales_pack.dtype = llm_engine::DataType::FP16;
    w.scales_pack.owns_data = false;
    w.zeros_pack.data = owned->zeros.data();
    w.zeros_pack.dtype = llm_engine::DataType::INT8;
    w.zeros_pack.owns_data = false;
    return true;
}

double elapsed_ms(const std::chrono::steady_clock::time_point& begin) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
}

std::vector<int> parse_rows(const std::string& value) {
    std::vector<int> rows;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ',')) {
        int row = std::stoi(part);
        if (row >= 1 && row <= 8) rows.push_back(row);
    }
    return rows;
}

} // namespace

int main(int argc, char** argv) {
    std::string weights_dir = "../weights";
    std::vector<int> row_values{1, 2, 3, 4, 5, 6, 7, 8};
    int warmup = 1;
    int repeats = 3;
    int threads = 4;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights-dir" && i + 1 < argc) weights_dir = argv[++i];
        else if (arg == "--rows" && i + 1 < argc) row_values = parse_rows(argv[++i]);
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::max(0, std::stoi(argv[++i]));
        else if (arg == "--repeats" && i + 1 < argc) repeats = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--threads" && i + 1 < argc) threads = std::max(1, std::stoi(argv[++i]));
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--weights-dir DIR] [--rows 2,4,8]"
                      << " [--warmup N] [--repeats N] [--threads N]" << std::endl;
            return 2;
        }
    }

    llm_engine::ThreadPool pool(threads);
    llm_engine::g_thread_pool = &pool;
    const OpSpec ops[] = {
        {"q_proj", "layer0_q_proj", 1536, 1536, false},
        {"o_proj", "layer0_o_proj", 1536, 1536, false},
        {"gate_proj", "layer0_gate_proj", 1536, 8960, false},
        {"up_proj", "layer0_up_proj", 1536, 8960, false},
        {"down_proj", "layer0_down_proj", 8960, 1536, false},
        {"lm_head_argmax", "lm_head", 1536, 151936, true},
    };

    std::cout << "op,B,K,N,row_loop_ms,true_batch_ms,speedup,row_loop_tokens_per_second,"
                 "true_batch_tokens_per_second,max_abs,argmax_match,kernel_calls,rows_total,"
                 "panel_tasks,row_gemv_fallbacks,full_logits_elements_written\n";
    for (const OpSpec& op : ops) {
        OwnedWeight owned;
        if (!load_weight(weights_dir, op, &owned)) return 1;
        for (int B : row_values) {
            std::vector<fp16_t> x((size_t)B * op.K);
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = (fp16_t)(0.15f * std::sin((float)(i % 4096) * 0.013f));
            }
            size_t output_elements = op.argmax ? 0 : (size_t)B * op.N;
            std::vector<fp16_t> reference(output_elements);
            std::vector<fp16_t> batch(output_elements);
            std::vector<ArgmaxResult> ref_argmax((size_t)B);
            std::vector<ArgmaxResult> batch_argmax((size_t)B);
            size_t argmax_bytes =
                llm_engine::arm_neon::linear_gptq_int8_batch_argmax_workspace_bytes(B, owned.weight);
            std::vector<uint8_t> argmax_workspace(std::max<size_t>(argmax_bytes, 64));

            auto run_reference = [&]() {
                for (int row = 0; row < B; ++row) {
                    if (op.argmax) {
                        ref_argmax[(size_t)row] =
                            llm_engine::arm_neon::linear_gptq_int8_decode_argmax_neon(
                                x.data() + (size_t)row * op.K, owned.weight, nullptr, 0);
                    } else {
                        llm_engine::arm_neon::linear_gptq_int8_decode_neon(
                            x.data() + (size_t)row * op.K, owned.weight,
                            reference.data() + (size_t)row * op.N, nullptr, nullptr, 0);
                    }
                }
            };
            auto run_batch = [&]() {
                if (op.argmax) {
                    if (B == 1) {
                        batch_argmax[0] =
                            llm_engine::arm_neon::linear_gptq_int8_decode_argmax_neon(
                                x.data(), owned.weight, nullptr, 0);
                    } else {
                        llm_engine::arm_neon::linear_gptq_int8_decode_argmax_batch_neon(
                            x.data(), B, owned.weight, batch_argmax.data(),
                            argmax_workspace.data(), argmax_workspace.size());
                    }
                } else {
                    llm_engine::arm_neon::linear_gptq_int8_batch_neon(
                        x.data(), B, owned.weight, batch.data(), nullptr, nullptr, 0);
                }
            };

            for (int i = 0; i < warmup; ++i) {
                run_reference();
                run_batch();
            }
            auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i) run_reference();
            double row_ms = elapsed_ms(begin) / repeats;
            GPTQBatchKernelStats before =
                llm_engine::arm_neon::snapshot_gptq_batch_kernel_stats();
            begin = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i) run_batch();
            double batch_ms = elapsed_ms(begin) / repeats;
            GPTQBatchKernelStats counters = llm_engine::arm_neon::diff_gptq_batch_kernel_stats(
                before, llm_engine::arm_neon::snapshot_gptq_batch_kernel_stats());

            float max_abs = 0.0f;
            bool argmax_match = true;
            if (op.argmax) {
                run_reference();
                run_batch();
                for (int row = 0; row < B; ++row) {
                    argmax_match = argmax_match &&
                        ref_argmax[(size_t)row].index == batch_argmax[(size_t)row].index;
                }
            } else {
                for (size_t i = 0; i < batch.size(); ++i) {
                    max_abs = std::max(max_abs,
                        std::fabs((float)batch[i] - (float)reference[i]));
                }
            }
            std::cout << op.name << ',' << B << ',' << op.K << ',' << op.N << ','
                      << row_ms << ',' << batch_ms << ',' << (row_ms / batch_ms) << ','
                      << (B * 1000.0 / row_ms) << ',' << (B * 1000.0 / batch_ms) << ','
                      << max_abs << ',' << (argmax_match ? 1 : 0) << ','
                      << counters.kernel_calls << ',' << counters.rows_total << ','
                      << counters.output_panel_tasks << ',' << counters.row_gemv_fallbacks << ','
                      << counters.full_logits_elements_written << '\n';
        }
    }
    llm_engine::g_thread_pool = nullptr;
    return 0;
}
