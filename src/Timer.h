#pragma once

class Timer 
{
    public:
        Timer();
        ~Timer();
        void stop();
        double getElapsedSeconds() const;
        double getElapsedMilliseconds() const;
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> m_startTimePoint;
};
