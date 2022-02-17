#pragma once
#include <functional>
#include <deque>


class FunctionQueue
{
public:
    void AddFunction(std::function<void()>&& func) 
    {
        m_funcQueue.push_back(func);
    }
    
    void Flush()
    {
        for (auto it = m_funcQueue.rbegin(); it != m_funcQueue.rend(); it++) {
            (*it)();
        }
        m_funcQueue.clear();
    }

private:
    std::deque<std::function<void()>> m_funcQueue;
};