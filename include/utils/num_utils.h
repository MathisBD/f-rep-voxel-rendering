#pragma once
#include <assert.h>
#include <stdint.h>


class NumUtils
{
public:
    // Rounds n upwards towards the closest multiple of k.
    static uint64_t RoundUpToMultiple(uint64_t n, uint64_t k) 
    {
        return ((n + k - 1) / k) * k;
    }
};