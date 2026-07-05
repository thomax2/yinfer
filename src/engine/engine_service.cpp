#include "llm_engine/engine/engine_service.h"

#include <cstdlib>
#include <exception>
#include <future>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace llm_engine {

namespace {

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

enum class EngineCommandType {
    SUBMIT,
    ABORT,
    CLEAR_SESSION,
    CLEAR_HISTORY,
    STOP
};

} // namespace

struct EngineService::EngineCommand {
    EngineCommandType type = EngineCommandType::SUBMIT;
    RequestId request_id = 0;
    SessionId session_id = 0;
    std::vector<int> prompt_tokens;
    SamplingParams sampling;
    TokenCallback callback;
    std::shared_ptr<std::promise<RequestId>> submit_promise;
    std::shared_ptr<std::promise<void>> done_promise;
};

EngineService::EngineService(LLMEngine& engine) : engine_(engine) {
    if (!engine_.scheduler_enabled()) {
        std::cerr << "[SERVICE] EngineService requires LLM_ENABLE_SCHEDULER=1"
                  << std::endl;
    }
}

EngineService::~EngineService() {
    stop();
}

void EngineService::start() {
    if (running_.load()) {
        return;
    }
    if (!engine_.scheduler_enabled()) {
        throw std::runtime_error("EngineService requires LLM_ENABLE_SCHEDULER=1");
    }

    stop_requested_.store(false);
    running_.store(true);
    worker_ = std::thread(&EngineService::engine_loop, this);
    if (debug_enabled()) {
        std::cerr << "[SERVICE] start" << std::endl;
    }
}

void EngineService::stop() {
    if (!running_.load() && !worker_.joinable()) {
        return;
    }

    stop_requested_.store(true);
    if (worker_.joinable() && std::this_thread::get_id() != worker_.get_id()) {
        auto command = std::make_unique<EngineCommand>();
        command->type = EngineCommandType::STOP;
        {
            std::lock_guard<std::mutex> lk(mu_);
            commands_.push_back(std::move(command));
        }
        cv_.notify_one();
        if (debug_enabled()) {
            std::cerr << "[SERVICE] stop requested" << std::endl;
        }
        worker_.join();
        if (debug_enabled()) {
            std::cerr << "[SERVICE] worker joined" << std::endl;
        }
    }

    running_.store(false);
    status_cv_.notify_all();
}

RequestId EngineService::submit(
    const std::vector<int>& prompt_tokens,
    const SamplingParams& sampling,
    TokenCallback callback
) {
    return submit(1, prompt_tokens, sampling, std::move(callback));
}

RequestId EngineService::submit(
    SessionId session_id,
    const std::vector<int>& prompt_tokens,
    const SamplingParams& sampling,
    TokenCallback callback
) {
    if (!running_.load() || stop_requested_.load()) {
        throw std::runtime_error("EngineService is not running");
    }

    auto promise = std::make_shared<std::promise<RequestId>>();
    auto future = promise->get_future();
    auto command = std::make_unique<EngineCommand>();
    command->type = EngineCommandType::SUBMIT;
    command->session_id = session_id;
    command->prompt_tokens = prompt_tokens;
    command->sampling = sampling;
    command->callback = std::move(callback);
    command->submit_promise = promise;

    if (debug_enabled()) {
        std::cerr << "[SERVICE] enqueue submit"
                  << " session=" << session_id
                  << " prompt_tokens=" << command->prompt_tokens.size()
                  << std::endl;
    }

    enqueue_command(std::move(command));
    RequestId id = future.get();
    if (debug_enabled()) {
        std::cerr << "[SERVICE] returned request_id=" << id << std::endl;
    }
    return id;
}

void EngineService::abort(RequestId id) {
    auto command = std::make_unique<EngineCommand>();
    command->type = EngineCommandType::ABORT;
    command->request_id = id;
    enqueue_void_command(std::move(command));
}

void EngineService::clear_session(SessionId session_id) {
    auto command = std::make_unique<EngineCommand>();
    command->type = EngineCommandType::CLEAR_SESSION;
    command->session_id = session_id;
    enqueue_void_command(std::move(command));
}

void EngineService::clear_history() {
    auto command = std::make_unique<EngineCommand>();
    command->type = EngineCommandType::CLEAR_HISTORY;
    enqueue_void_command(std::move(command));
}

bool EngineService::wait_until_finished(RequestId id) {
    std::unique_lock<std::mutex> lk(status_mu_);
    status_cv_.wait(lk, [&] {
        auto it = snapshots_.find(id);
        if (it != snapshots_.end() && is_terminal(it->second.status)) {
            return true;
        }
        return !running_.load() && stop_requested_.load();
    });
    auto it = snapshots_.find(id);
    return it != snapshots_.end() && is_terminal(it->second.status);
}

bool EngineService::request_snapshot(RequestId id, EngineRequestSnapshot* out) const {
    if (!out) {
        return false;
    }
    std::lock_guard<std::mutex> lk(status_mu_);
    auto it = snapshots_.find(id);
    if (it == snapshots_.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

bool EngineService::is_running() const {
    return running_.load();
}

void EngineService::engine_loop() {
    if (debug_enabled()) {
        std::cerr << "[SERVICE] engine_loop enter" << std::endl;
    }

    while (true) {
        bool did_work = false;
        std::vector<std::unique_ptr<EngineCommand>> commands;

        {
            std::unique_lock<std::mutex> lk(mu_);
            if (commands_.empty() && !engine_.has_pending_requests() && !stop_requested_.load()) {
                cv_.wait(lk, [&] {
                    return stop_requested_.load() ||
                           !commands_.empty() ||
                           engine_.has_pending_requests();
                });
            }

            while (!commands_.empty()) {
                commands.push_back(std::move(commands_.front()));
                commands_.pop_front();
            }
        }

        for (auto& command : commands) {
            handle_command(*command);
            did_work = true;
        }

        if (stop_requested_.load()) {
            break;
        }

        if (engine_.has_pending_requests()) {
            if (debug_enabled()) {
                std::cerr << "[SERVICE] step active_or_pending=1" << std::endl;
            }
            engine_.step_once();
            refresh_all_tracked_statuses();
            did_work = true;
        }

        if (!did_work) {
            std::this_thread::yield();
        }
    }

    running_.store(false);
    status_cv_.notify_all();
    {
        std::lock_guard<std::mutex> lk(mu_);
        while (!commands_.empty()) {
            std::unique_ptr<EngineCommand> pending = std::move(commands_.front());
            commands_.pop_front();
            try {
                throw std::runtime_error("EngineService stopped before command was handled");
            } catch (...) {
                if (pending->submit_promise) {
                    pending->submit_promise->set_exception(std::current_exception());
                }
                if (pending->done_promise) {
                    pending->done_promise->set_exception(std::current_exception());
                }
            }
        }
    }
    if (debug_enabled()) {
        std::cerr << "[SERVICE] engine_loop exit" << std::endl;
    }
}

void EngineService::handle_command(EngineCommand& command) {
    try {
        switch (command.type) {
            case EngineCommandType::SUBMIT: {
                RequestId id = engine_.submit_async(
                    command.session_id,
                    command.prompt_tokens,
                    command.sampling,
                    std::move(command.callback));
                refresh_request_status(id);
                if (command.submit_promise) {
                    command.submit_promise->set_value(id);
                }
                if (debug_enabled()) {
                    std::cerr << "[SERVICE] handle submit id=" << id << std::endl;
                }
                break;
            }
            case EngineCommandType::ABORT:
                engine_.abort(command.request_id);
                refresh_request_status(command.request_id);
                if (command.done_promise) {
                    command.done_promise->set_value();
                }
                if (debug_enabled()) {
                    std::cerr << "[SERVICE] handle abort id=" << command.request_id << std::endl;
                }
                break;
            case EngineCommandType::CLEAR_SESSION:
                engine_.clear_session(command.session_id);
                if (command.done_promise) {
                    command.done_promise->set_value();
                }
                if (debug_enabled()) {
                    std::cerr << "[SERVICE] handle clear_session"
                              << " session=" << command.session_id
                              << std::endl;
                }
                break;
            case EngineCommandType::CLEAR_HISTORY:
                engine_.clear_history();
                if (command.done_promise) {
                    command.done_promise->set_value();
                }
                if (debug_enabled()) {
                    std::cerr << "[SERVICE] handle clear_history" << std::endl;
                }
                break;
            case EngineCommandType::STOP:
                stop_requested_.store(true);
                if (command.done_promise) {
                    command.done_promise->set_value();
                }
                if (debug_enabled()) {
                    std::cerr << "[SERVICE] handle stop" << std::endl;
                }
                break;
        }
    } catch (...) {
        if (command.submit_promise) {
            command.submit_promise->set_exception(std::current_exception());
        }
        if (command.done_promise) {
            command.done_promise->set_exception(std::current_exception());
        }
    }
}

void EngineService::enqueue_command(std::unique_ptr<EngineCommand> command) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        commands_.push_back(std::move(command));
    }
    cv_.notify_one();
}

void EngineService::enqueue_void_command(std::unique_ptr<EngineCommand> command) {
    if (!running_.load() || stop_requested_.load()) {
        throw std::runtime_error("EngineService is not running");
    }
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();
    command->done_promise = promise;
    enqueue_command(std::move(command));
    future.get();
}

void EngineService::refresh_request_status(RequestId id) {
    const RequestState* request = engine_.get_request(id);
    EngineRequestSnapshot snapshot;
    snapshot.exists = request != nullptr;
    snapshot.request_id = id;
    if (request) {
        snapshot.session_id = request->session_id;
        snapshot.status = request->status;
        snapshot.error_message = request->error_message;
        snapshot.token_count = request->token_count;
        snapshot.num_generated_tokens = request->num_generated_tokens;
        snapshot.metrics = request->metrics;
    }

    {
        std::lock_guard<std::mutex> lk(status_mu_);
        snapshots_[id] = snapshot;
    }
    status_cv_.notify_all();
}

void EngineService::refresh_all_tracked_statuses() {
    std::vector<RequestId> ids;
    {
        std::lock_guard<std::mutex> lk(status_mu_);
        ids.reserve(snapshots_.size());
        for (const auto& kv : snapshots_) {
            ids.push_back(kv.first);
        }
    }
    for (RequestId id : ids) {
        refresh_request_status(id);
    }
}

bool EngineService::debug_enabled() const {
    return env_flag("LLM_DEBUG_SERVICE");
}

bool EngineService::is_terminal(RequestStatus status) const {
    return status == RequestStatus::FINISHED ||
           status == RequestStatus::ABORTED ||
           status == RequestStatus::FAILED;
}

const char* EngineService::request_status_name(RequestStatus status) const {
    switch (status) {
        case RequestStatus::WAITING:
            return "WAITING";
        case RequestStatus::RUNNING_PREFILL:
            return "RUNNING_PREFILL";
        case RequestStatus::RUNNING_DECODE:
            return "RUNNING_DECODE";
        case RequestStatus::FINISHED:
            return "FINISHED";
        case RequestStatus::ABORTED:
            return "ABORTED";
        case RequestStatus::FAILED:
            return "FAILED";
    }
    return "UNKNOWN";
}

} // namespace llm_engine
