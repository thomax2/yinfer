#pragma once

#include "llm_engine/tensor.h"
#include "llm_engine/status.h"

namespace llm_engine {
namespace reference {

Status add_ref(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
);

Status matmul_ref(
    const Tensor& A,
    const Tensor& B,
    Tensor& C
);

} // reference
} // llm_engine