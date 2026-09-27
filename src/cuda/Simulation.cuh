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

void launchTestColor(float4* color, float seconds, cudaStream_t stream);
