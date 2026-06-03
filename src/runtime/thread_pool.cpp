#include "llm_engine/runtime/thread_pool.h"

#include <algorithm>

namespace llm_engine {

ThreadPool* g_thread_pool = nullptr;

ThreadPool::ThreadPool(int num_threads) {
    if (num_threads < 1) num_threads = 1;
    num_threads_ = num_threads;

    if (num_threads_ <= 1) {
        // 单线程模式：不创建任何 worker，parallel_for 会走同步路径
        return;
    }

    workers_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i) {
        workers_.emplace_back([this] { this->worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& th : workers_) {
        if (th.joinable()) th.join();
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = tasks_.front();
            tasks_.pop();
        }

        // 执行任务
        (*task.fn)(task.begin, task.end);

        // 完成后递减计数，必要时通知 parallel_for
        if (task.remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(*task.done_mu);
            task.done_cv->notify_all();
        }
    }
}

void ThreadPool::parallel_for(int begin,
                              int end,
                              int grain,
                              const RangeFn& fn) {
    if (end <= begin) return;
    if (grain < 1) grain = 1;

    int total = end - begin;

    // 单线程或任务过小，直接同步执行
    if (num_threads_ <= 1 || total <= grain) {
        fn(begin, end);
        return;
    }

    // 切成尽可能均匀的若干段；每段不小于 grain
    int max_chunks = (total + grain - 1) / grain;
    int n_chunks = std::min(num_threads_, max_chunks);
    if (n_chunks <= 1) {
        fn(begin, end);
        return;
    }

    // 让 chunk 至少是 grain 的整数倍，保持每个 worker 有完整的最小粒度
    int base_units = total / n_chunks;
    int remainder = total % n_chunks;

    std::atomic<int> remaining(n_chunks);
    std::mutex done_mu;
    std::condition_variable done_cv;

    {
        std::lock_guard<std::mutex> lk(mu_);
        int cursor = begin;
        for (int i = 0; i < n_chunks; ++i) {
            int len = base_units + (i < remainder ? 1 : 0);
            Task t;
            t.begin = cursor;
            t.end = cursor + len;
            t.fn = &fn;
            t.remaining = &remaining;
            t.done_mu = &done_mu;
            t.done_cv = &done_cv;
            tasks_.push(t);
            cursor += len;
        }
    }
    cv_.notify_all();

    {
        std::unique_lock<std::mutex> lk(done_mu);
        done_cv.wait(lk, [&] {
            return remaining.load(std::memory_order_acquire) == 0;
        });
    }
}

} // namespace llm_engine
