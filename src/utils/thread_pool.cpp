#include "utils/thread_pool.h"
#include <stdio.h>
#include <assert.h>


ThreadPool::ThreadPool(size_t threadCount) : m_stop(false) 
{
    for (size_t i = 0; i < threadCount; i++) {
        m_threads.emplace_back(&ThreadPool::Worker, this);
    }
}

ThreadPool::~ThreadPool() 
{
    if (!m_stop) {
        Stop();   
    } 
}

void ThreadPool::Worker() 
{
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock lock(m_mutex);
            m_onEvent.wait(lock, [&]() { return m_stop || !m_taskQueue.empty(); });
            
            if (m_stop) {
                return;
            }
            assert(!m_taskQueue.empty());
            task = std::move(m_taskQueue.front());
            m_taskQueue.pop();
        }
        task();
    }    
}


void ThreadPool::Stop() 
{
    {
        std::lock_guard lock(m_mutex);
        m_stop = true;
    }
    m_onEvent.notify_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
        std::lock_guard lock(m_mutex);
        for (auto& t : m_threads) {
            t.join();
        }
        m_threads.clear();
    }
}