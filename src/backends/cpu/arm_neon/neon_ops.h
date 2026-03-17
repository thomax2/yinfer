#pragma once

#include "llm_engine/tensor.h"
#include "llm_engine/status.h"

namespace llm_engine {
namespace arm_neon {

Status matmul_neon(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float* workspace
);

}
}