#pragma once

#include "llm_engine/tensor.h"

namespace llm_engine {
namespace arm_neon {

enum class GPTQPackKind {
    W8A16_FP16_PANEL,
    W8A8_SDOT_PANEL
};

struct GPTQInt8Weight {
    int K = 0;
    int N = 0;
    int group_size = 128;
    int num_groups = 0;
    bool has_zero = true;
    bool has_g_idx = false;

    GPTQPackKind pack_kind = GPTQPackKind::W8A16_FP16_PANEL;

    Tensor qweight_pack;  // INT8, [ceil(N/16), align_up(K, 8), 16]
    Tensor scales_pack;   // FP16, [ceil(N/16), num_groups, 16]
    Tensor zeros_pack;    // INT8, [ceil(N/16), num_groups, 16]
    Tensor g_idx;         // INT32, optional [K]
};

inline int align_up_int(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace arm_neon
} // namespace llm_engine
