#pragma once

namespace llm_engine {
namespace arm_neon {

// Legacy FP32 blocking kept for old tests and reference paths.
constexpr int MR = 8;
constexpr int NR = 12;

constexpr int MR_F16 = 8;
constexpr int NR_F16 = 16;

constexpr int MR_I8 = 4;
constexpr int NR_I8 = 16;
constexpr int KR_I8 = 4;

constexpr int KC = 256;
constexpr int MC = 256;
constexpr int NC = 256;

}
}
