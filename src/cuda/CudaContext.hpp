#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>

class CudaContext
{
public:
    explicit CudaContext(
        VkPhysicalDevice physicalDevice
    );

    ~CudaContext();

    [[nodiscard]]
    cudaStream_t stream() const noexcept
    {
        return m_stream;
    }

    [[nodiscard]]
    int deviceIndex() const noexcept
    {
        return m_deviceIndex;
    }

    void writeTestColor(void* pixels, uint32_t width, uint32_t height,
                        bool bgra, cudaExternalSemaphore_t ready, float seconds);

private:
    int findMatchingDevice(
        VkPhysicalDevice physicalDevice
    );

private:
    int m_deviceIndex = -1;

    cudaStream_t m_stream = nullptr;
};
