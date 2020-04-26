#include <chrono>

#include "Timer.h"

Timer::Timer() 
    : m_startTimePoint(std::chrono::high_resolution_clock::now()) 
{
    // Starts the timer
}

double Timer::getElapsedTime(const Timer::UnitDivider t) const
{
    const std::chrono::time_point<std::chrono::high_resolution_clock> currentTimePoint = 
        std::chrono::high_resolution_clock::now();

    const long long start = 
        std::chrono::time_point_cast<std::chrono::microseconds>(m_startTimePoint).time_since_epoch().count();

    const long long end = 
        std::chrono::time_point_cast<std::chrono::microseconds>(currentTimePoint).time_since_epoch().count();

    return (double) (end - start) / (int) t;
}
