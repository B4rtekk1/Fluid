#include "Simulation.cuh"

#include <cmath>

__global__ void testColorKernel(float4* color, float seconds)
{
    color[0] = make_float4(
        0.5f + 0.5f * sinf(seconds),
        0.5f + 0.5f * sinf(seconds + 2.0943951f),
        0.5f + 0.5f * sinf(seconds + 4.1887902f),
        1.0f);
}

void launchTestColor(float4* color, float seconds, cudaStream_t stream)
{
    testColorKernel<<<1, 1, 0, stream>>>(color, seconds);
}

__global__
void integrateKernel(
    float4* positions,
    float4* velocities,
    uint32_t count,
    float dt)
{
    const uint32_t i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i >= count)
        return;

    float4 p = positions[i];
    float4 v = velocities[i];

    v.y -= 9.81f * dt;

    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;

    constexpr float floorY = -1.0f;

    if (p.y < floorY)
    {
        p.y = floorY;

        if (v.y < 0.0f)
            v.y *= -0.5f;
    }

    positions[i] = p;
    velocities[i] = v;
}

void launchIntegrate(
    float4* positions,
    float4* velocities,
    uint32_t particleCount,
    float dt,
    cudaStream_t stream)
{
    constexpr uint32_t blockSize = 256;

    const uint32_t blockCount =
        (particleCount +
         blockSize - 1) /
        blockSize;

    integrateKernel<<<
        blockCount,
        blockSize,
        0,
        stream
    >>>(
        positions,
        velocities,
        particleCount,
        dt
    );
}
