#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace llm_engine {

class ThreadPool {
public:
    using RangeFn = std::function<void(int, int)>;

    explicit ThreadPool(int num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int num_threads() const { return num_threads_; }

    // 在 [begin, end) 上做并行 for。
    // 内部把范围切成若干 [task_begin, task_end)，每段大小不小于 grain，
    // worker 调用 fn(task_begin, task_end)。
    // 必须等所有任务完成后才返回。
    // 如果 num_threads <= 1 或任务量过小，会直接在调用线程同步执行。
    void parallel_for(int begin,
                      int end,
                      int grain,
                      const RangeFn& fn);

private:
    struct Task {
        int begin;
        int end;
        const RangeFn* fn;        // 指向 parallel_for 栈上对象，生命周期由 parallel_for 保证
        std::atomic<int>* remaining; // 同上
        std::mutex* done_mu;
        std::condition_variable* done_cv;
    };

    void worker_loop();

    int num_threads_;
    std::vector<std::thread> workers_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<Task> tasks_;
    bool stop_ = false;
};

extern ThreadPool* g_thread_pool;

} // namespace llm_engine
