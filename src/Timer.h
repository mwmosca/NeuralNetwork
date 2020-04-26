#pragma once

#include <chrono>

class Timer 
{
    // This class uses timers to measure performance. The timer is started by the constructor. 

private:
    const std::chrono::time_point<std::chrono::high_resolution_clock> m_startTimePoint;

public:
    Timer();
    
    ~Timer() = default;

    Timer(const Timer&)            = delete;
    Timer(Timer&&)                 = delete;
    Timer& operator=(const Timer&) = delete;
    Timer& operator=(Timer&&)      = delete;

    enum class UnitDivider    // Used to convert microsecond values
    {
        sec = 1000000, msec = 1000, usec = 1
    };
    
    double getElapsedTime(const Timer::UnitDivider t) const;
};
