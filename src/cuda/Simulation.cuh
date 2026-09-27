#pragma once

#include <cuda_runtime.h>
#include <cstdint>

void launchIntegrate(
    float4* positions,
    float4* velocities,
    uint32_t particleCount,
    float dt,
    cudaStream_t stream
);

void launchTestColor(uint32_t* pixels, uint32_t width, uint32_t height,
                     bool bgra, float seconds, cudaStream_t stream);
