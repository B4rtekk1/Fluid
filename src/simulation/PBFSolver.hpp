#pragma once

#include <cuda_runtime.h>

#include <cstdint>

class PBFSolver
{
public:
    PBFSolver(
        uint32_t particleCount,
        cudaStream_t stream
    );

    void setBuffers(
        float4* positions,
        float4* velocities
    );

    void simulate(float dt);

private:
    uint32_t m_particleCount = 0;

    cudaStream_t m_stream = nullptr;

    float4* m_positions = nullptr;
    float4* m_velocities = nullptr;
};