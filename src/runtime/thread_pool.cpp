#include "llm_engine/runtime/thread_pool.h"

#include <algorithm>

namespace llm_engine {

ThreadPool* g_thread_pool = nullptr;

ThreadPool::ThreadPool(int num_threads) {
    if (num_threads < 1) num_threads = 1;
    num_threads_ = num_threads;

    // 语义：num_threads_ 是“总计算线程数”，包含主线程。
    // 因此实际创建的 worker 数 = num_threads_ - 1。
    // ThreadPool(1)：worker 数 = 0，parallel_for 由主线程同步执行。
    int worker_count = std::max(0, num_threads_ - 1);
    if (worker_count == 0) {
        return;
    }

    workers_.reserve(worker_count);
    for (int i = 0; i < worker_count; ++i) {
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

        // 完成后递减计数，必要时通知 parallel_for。
        // 必须在持有 done_mu 的情况下递减 remaining：否则一旦 remaining 归零，
        // 主线程的 wait 谓词可能立即返回并退出 parallel_for，
        // 导致栈上的 done_mu / done_cv 被析构，
        // worker 后续再去 lock done_mu 就会访问已析构对象。
        {
            std::lock_guard<std::mutex> lk(*task.done_mu);
            if (task.remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                task.done_cv->notify_all();
            }
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

    // 单线程或任务过小，直接由主线程同步执行
    if (num_threads_ <= 1 || total <= grain) {
        fn(begin, end);
        return;
    }

    // n_chunks 不超过总线程数（含主线程），也不超过粒度允许的最大切分数
    int max_chunks = (total + grain - 1) / grain;
    int n_chunks = std::min(num_threads_, max_chunks);
    if (n_chunks <= 1) {
        fn(begin, end);
        return;
    }

    // 切片：尽量均匀
    std::vector<std::pair<int, int>> ranges;
    ranges.reserve(n_chunks);

    int base = total / n_chunks;
    int rem = total % n_chunks;

    int cursor = begin;
    for (int i = 0; i < n_chunks; ++i) {
        int len = base + (i < rem ? 1 : 0);
        ranges.push_back({cursor, cursor + len});
        cursor += len;
    }

    // 主线程负责 chunk 0；剩下的交给 worker。
    int worker_tasks = n_chunks - 1;

    std::atomic<int> remaining(worker_tasks);
    std::mutex done_mu;
    std::condition_variable done_cv;

    if (worker_tasks > 0) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 1; i < n_chunks; ++i) {
                Task t;
                t.begin = ranges[i].first;
                t.end = ranges[i].second;
                t.fn = &fn;
                t.remaining = &remaining;
                t.done_mu = &done_mu;
                t.done_cv = &done_cv;
                tasks_.push(t);
            }
        }
        cv_.notify_all();
    }

    // 主线程立即参与计算，执行 chunk 0
    fn(ranges[0].first, ranges[0].second);

    // 等 worker 完成；用 remaining.load() 作为谓词，避免错过 notify
    if (worker_tasks > 0) {
        std::unique_lock<std::mutex> lk(done_mu);
        done_cv.wait(lk, [&] {
            return remaining.load(std::memory_order_acquire) == 0;
        });
    }
}

} // namespace llm_engine
