#include "PBFSolver.hpp"

#include "cuda/Simulation.cuh"

PBFSolver::PBFSolver(
    uint32_t particleCount,
    cudaStream_t stream)
    :
    m_particleCount(particleCount),
    m_stream(stream)
{
}

void PBFSolver::setBuffers(
    float4* positions,
    float4* velocities)
{
    m_positions = positions;
    m_velocities = velocities;
}

void PBFSolver::simulate(float dt)
{
    launchIntegrate(
        m_positions,
        m_velocities,
        m_particleCount,
        dt,
        m_stream
    );
}