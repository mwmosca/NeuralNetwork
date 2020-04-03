#include <chrono>
#include <iostream>

#include "Timer.h"

Timer::Timer() 
{
    m_startTimePoint = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() 
{
    stop();
}

void Timer::stop()
{
    std::chrono::time_point<std::chrono::high_resolution_clock> endTimePoint = std::chrono::high_resolution_clock::now();
    int64_t start = std::chrono::time_point_cast<std::chrono::microseconds> (m_startTimePoint).time_since_epoch().count();
    int64_t end = std::chrono::time_point_cast<std::chrono::microseconds> (endTimePoint).time_since_epoch().count();
    long duration = end - start;        // microseconds
    double ms = duration * 0.001;       // milliseconds
    double s = duration * 0.000001;     // seconds
    
    // Output the time span to the console at the end of scope
    // std::cout << s << " seconds" << std::endl;    
}

double Timer::getElapsedSeconds() const
{
    std::chrono::time_point<std::chrono::high_resolution_clock> currentTimePoint = std::chrono::high_resolution_clock::now();
    int64_t start = std::chrono::time_point_cast<std::chrono::microseconds> (m_startTimePoint).time_since_epoch().count();
    int64_t end = std::chrono::time_point_cast<std::chrono::microseconds> (currentTimePoint).time_since_epoch().count();
    return (end - start) * 0.000001;
}

double Timer::getElapsedMilliseconds() const 
{
    std::chrono::time_point<std::chrono::high_resolution_clock> currentTimePoint = std::chrono::high_resolution_clock::now();
    int64_t start = std::chrono::time_point_cast<std::chrono::microseconds> (m_startTimePoint).time_since_epoch().count();
    int64_t end = std::chrono::time_point_cast<std::chrono::microseconds> (currentTimePoint).time_since_epoch().count();
    return (end - start) * 0.001;
}
