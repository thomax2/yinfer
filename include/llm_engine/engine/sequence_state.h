#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "llm_engine/cache/hash.h"

namespace llm_engine {

using SessionId = uint64_t;

enum class SequenceStatus {
    IDLE,
    RUNNING,
    FINISHED,
    ABORTED,
    FAILED
};

struct SequenceState {
    SessionId session_id = 0;

    std::vector<int> all_tokens;
    std::vector<int> generated_tokens;

    int history_pos = 0;
    int max_written_pos = -1;
    std::vector<int> block_table;
    int num_computed_tokens = 0;
    int cached_prefix_tokens = 0;
    int cached_prefix_blocks = 0;
    HashValue last_prefix_hash;

    SequenceStatus status = SequenceStatus::IDLE;
    std::string error_message;

    void reset() {
        all_tokens.clear();
        generated_tokens.clear();
        history_pos = 0;
        max_written_pos = -1;
        block_table.clear();
        num_computed_tokens = 0;
        cached_prefix_tokens = 0;
        cached_prefix_blocks = 0;
        last_prefix_hash = {};
        status = SequenceStatus::IDLE;
        error_message.clear();
    }
};

} // namespace llm_engine
