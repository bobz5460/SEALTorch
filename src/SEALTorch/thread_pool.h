#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace sealtorch
{
    class ThreadPool
    {
    public:
        explicit ThreadPool(std::size_t thread_count)
        {
            thread_count = std::max<std::size_t>(1, thread_count);
            workers_.reserve(thread_count);
            for (std::size_t index = 0; index < thread_count; ++index)
                workers_.emplace_back([this] { worker_loop(); });
        }

        ~ThreadPool()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            condition_.notify_all();
            for (std::thread &worker : workers_) worker.join();
        }

        void ensure_thread_count(std::size_t thread_count)
        {
            thread_count = std::max<std::size_t>(1, thread_count);
            std::lock_guard<std::mutex> lock(mutex_);
            workers_.reserve(thread_count);
            while (workers_.size() < thread_count)
                workers_.emplace_back([this] { worker_loop(); });
        }

        template <typename Function>
        void parallel_for_workers(std::size_t count, std::size_t requested, Function function)
        {
            if (count == 0) return;
            ensure_thread_count(std::min(count, std::max<std::size_t>(1, requested)));
            const std::size_t jobs = std::min({count, std::max<std::size_t>(1, requested), workers_.size()});
            std::vector<std::future<void>> futures;
            futures.reserve(jobs);
            for (std::size_t job = 0; job < jobs; ++job)
            {
                std::packaged_task<void()> task([&, job, jobs] { function(job, jobs); });
                futures.push_back(task.get_future());
                auto task_ptr = std::make_shared<std::packaged_task<void()>>(std::move(task));
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    queue_.emplace([task_ptr] { (*task_ptr)(); });
                }
                condition_.notify_one();
            }
            for (std::future<void> &future : futures) future.get();
        }

    private:
        void worker_loop()
        {
            for (;;)
            {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    condition_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                    if (stopping_ && queue_.empty()) return;
                    task = std::move(queue_.front());
                    queue_.pop();
                }
                task();
            }
        }

        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> queue_;
        std::mutex mutex_;
        std::condition_variable condition_;
        bool stopping_ = false;
    };
}
