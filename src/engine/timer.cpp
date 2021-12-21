#include "engine/timer.h"
#include <chrono>


std::chrono::high_resolution_clock::time_point Timer::s_startTime = std::chrono::high_resolution_clock::now();
float Timer::s_time = 0.0f;
float Timer::s_dt = 0.0f;

void Timer::UpdateTime()
{
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::duration<float, std::milli> duration;
    
    duration delta = clock::now() - s_startTime;
    float new_time = delta.count() / 1000.0f;

    s_dt = new_time - s_time;
    s_time = new_time;
}