#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <functional>
#include <future>


class ThreadPool
{
public:
    ThreadPool(size_t threadCount);
    ~ThreadPool();

    template <typename F>
    auto Enqueue(F task) -> std::future<decltype(task())>
    {
        using wrapper_type = std::packaged_task<decltype(task())()>;
        auto wrapper = std::make_shared<wrapper_type>(std::move(task));
        {
            std::lock_guard lock(m_mutex);
            m_taskQueue.emplace([=] {
                (*wrapper)();
            });
        }
        m_onEvent.notify_one();
        return wrapper->get_future();
    }
    
    void Stop();

private:
    std::mutex m_mutex;
    std::condition_variable m_onEvent;
    bool m_stop;

    std::vector<std::thread> m_threads;
    std::queue<std::function<void()>> m_taskQueue;

    void Start();
    
    void Worker();
};