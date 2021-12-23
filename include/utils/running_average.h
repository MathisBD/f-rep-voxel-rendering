#pragma once
#include <stdint.h>


// T should be some kind of numerical type (float, int, etc.)
template <typename T>
class RunningAverage
{
public:
    RunningAverage(uint32_t maxSamples)
    {
        m_maxSampleCount = maxSamples;
        m_currSampleCount = 0;
        m_nextSample = 0;

        m_samples = new T[m_maxSampleCount];
        m_average = 0;
    }

    ~RunningAverage() 
    {
        delete[] m_samples;
    }

    T GetAverage() const
    {
        return m_average;
    }

    void AddSample(T sample)
    {
        if (m_currSampleCount < m_maxSampleCount) {
            m_samples[m_nextSample] = sample;
            m_currSampleCount++;
            m_average = Average();    
        }
        else {
            m_average += (sample - m_samples[m_nextSample]) / m_maxSampleCount;
            m_samples[m_nextSample] = sample;
        }
        m_nextSample = (m_nextSample+1) % m_maxSampleCount;
    }

private:
    uint32_t m_maxSampleCount;
    uint32_t m_currSampleCount;
    uint32_t m_nextSample;
    T* m_samples;
    T m_average;

    T Average() const
    {
        T avg = 0;
        for (uint32_t i = 0; i < m_currSampleCount; i++) {
            avg += m_samples[i];
        }
        return avg / m_currSampleCount;
    }
};