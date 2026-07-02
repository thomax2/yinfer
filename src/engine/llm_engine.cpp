#include "llm_engine/engine/llm_engine.h"

#include "../../model/model.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

namespace llm_engine {

namespace {

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

} // namespace

LLMEngine::LLMEngine(QwenModel& model) : model_(model) {}

RequestId LLMEngine::submit(
    const std::vector<int>& prompt_tokens,
    const SamplingParams& sampling,
    TokenCallback callback
) {
    RequestId id = next_request_id_++;

    RequestState request;
    request.id = id;
    request.prompt_tokens = prompt_tokens;
    request.sampling = sampling;
    request.callback = std::move(callback);
    request.status = RequestStatus::RUNNING;

    auto inserted = requests_.emplace(id, std::move(request));
    RequestState& state = inserted.first->second;
    debug_log_submit(state);

    bool callback_stopped = false;
    try {
        model_.generate(
            state.prompt_tokens,
            state.sampling.max_new_tokens,
            [this, id, &callback_stopped](int token_id) {
                auto it = requests_.find(id);
                if (it == requests_.end()) {
                    callback_stopped = true;
                    return false;
                }

                RequestState& active = it->second;
                if (active.status == RequestStatus::ABORTED) {
                    callback_stopped = true;
                    return false;
                }

                active.generated_tokens.push_back(token_id);
                active.token_count++;

                if (active.callback && !active.callback(token_id)) {
                    callback_stopped = true;
                    return false;
                }
                return true;
            });

        state.status = callback_stopped ? RequestStatus::ABORTED : RequestStatus::FINISHED;
        debug_log_finished(state);
    } catch (const std::exception& e) {
        state.status = RequestStatus::FAILED;
        state.error_message = e.what();
        debug_log_failed(state);
    } catch (...) {
        state.status = RequestStatus::FAILED;
        state.error_message = "unknown exception";
        debug_log_failed(state);
    }

    state.callback = TokenCallback{};
    return id;
}

void LLMEngine::abort(RequestId id) {
    auto it = requests_.find(id);
    if (it != requests_.end() &&
        it->second.status != RequestStatus::FINISHED &&
        it->second.status != RequestStatus::FAILED) {
        it->second.status = RequestStatus::ABORTED;
    }
}

void LLMEngine::clear_history() {
    if (debug_enabled()) {
        std::cerr << "[ENGINE] clear_history" << std::endl;
    }
    model_.clear_history();
}

const RequestState* LLMEngine::get_request(RequestId id) const {
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool LLMEngine::debug_enabled() const {
    return env_flag("LLM_DEBUG_ENGINE");
}

void LLMEngine::debug_log_submit(const RequestState& request) const {
    if (!debug_enabled()) return;
    std::cerr << "[ENGINE] submit"
              << " id=" << request.id
              << " prompt_tokens=" << request.prompt_tokens.size()
              << " max_new_tokens=" << request.sampling.max_new_tokens
              << std::endl;
}

void LLMEngine::debug_log_finished(const RequestState& request) const {
    if (!debug_enabled()) return;
    const char* status = request.status == RequestStatus::ABORTED ? "aborted" : "finished";
    std::cerr << "[ENGINE] " << status
              << " id=" << request.id
              << " generated_tokens=" << request.generated_tokens.size()
              << std::endl;
}

void LLMEngine::debug_log_failed(const RequestState& request) const {
    if (!debug_enabled()) return;
    std::cerr << "[ENGINE] failed"
              << " id=" << request.id
              << " error=" << request.error_message
              << std::endl;
}

} // namespace llm_engine
