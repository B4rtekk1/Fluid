#include "Simulation.cuh"

#include <cmath>

__global__ void testColorKernel(uint32_t* pixels, uint32_t count, bool bgra, float seconds)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto channel = [](float phase) -> uint32_t {
        return static_cast<uint32_t>(255.0f * (0.5f + 0.5f * sinf(phase)) + 0.5f);
    };
    const uint32_t r = channel(seconds);
    const uint32_t g = channel(seconds + 2.0943951f);
    const uint32_t b = channel(seconds + 4.1887902f);
    pixels[i] = (bgra ? b | (g << 8) | (r << 16) : r | (g << 8) | (b << 16)) | 0xff000000u;
}

void launchTestColor(uint32_t* pixels, uint32_t width, uint32_t height,
                     bool bgra, float seconds, cudaStream_t stream)
{
    const uint32_t count = width * height;
    testColorKernel<<<(count + 255) / 256, 256, 0, stream>>>(pixels, count, bgra, seconds);
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
