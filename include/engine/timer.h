#pragma once
#include <chrono>

class Timer
{
private:
    static std::chrono::high_resolution_clock::time_point s_startTime;
public:
    // the absolute time since application start, in seconds.
    // updated every frame.
    static float s_time;
    // the time between the previous frame and the current frame, in seconds.
    // updated every frame.
    static float s_dt;
    // called from the application main loop
    static void UpdateTime();
};