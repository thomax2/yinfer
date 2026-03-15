#pragma once

namespace llm_engine {
namespace arm_neon {

// Register Blocking 大小
constexpr int MR = 8;
constexpr int NR = 12;

constexpr int KC = 256;
constexpr int MC = 256;
constexpr int NC = 256;

}
}