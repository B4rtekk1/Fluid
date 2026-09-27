#include "CudaContext.hpp"
#include "Simulation.cuh"

#include <cstring>
#include <stdexcept>

CudaContext::CudaContext(
    VkPhysicalDevice physicalDevice)
{
    m_deviceIndex =
        findMatchingDevice(physicalDevice);

    if (m_deviceIndex < 0)
    {
        throw std::runtime_error(
            "Couldn't match CUDA GPU with Vulkan GPU"
        );
    }

    if (cudaSetDevice(
            m_deviceIndex) != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaSetDevice failed"
        );
    }

    if (cudaStreamCreateWithFlags(
            &m_stream,
            cudaStreamNonBlocking) != cudaSuccess)
    {
        throw std::runtime_error(
            "cudaStreamCreate failed"
        );
    }

    const cudaError_t allocation = cudaMalloc(&m_deviceColor, sizeof(float4));
    if (allocation != cudaSuccess)
    {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
        throw std::runtime_error(std::string("cudaMalloc: ") + cudaGetErrorString(allocation));
    }
}

CudaContext::~CudaContext()
{
    if (m_deviceColor) cudaFree(m_deviceColor);
    if (m_stream)
    {
        cudaStreamDestroy(m_stream);
    }
}

int CudaContext::findMatchingDevice(
    VkPhysicalDevice physicalDevice)
{
    VkPhysicalDeviceIDProperties idProperties{
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES
    };

    VkPhysicalDeviceProperties2 properties{
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,

        .pNext =
            &idProperties
    };

    vkGetPhysicalDeviceProperties2(
        physicalDevice,
        &properties
    );

    int cudaDeviceCount = 0;

    cudaGetDeviceCount(
        &cudaDeviceCount
    );

    for (int i = 0;
         i < cudaDeviceCount;
         ++i)
    {
        cudaDeviceProp cudaProperties{};

        cudaGetDeviceProperties(
            &cudaProperties,
            i
        );

        if (std::memcmp(
                cudaProperties.uuid.bytes,
                idProperties.deviceUUID,
                VK_UUID_SIZE) == 0)
        {
            return i;
        }
    }

    return -1;
}

float4 CudaContext::testColor(float seconds)
{
    launchTestColor(m_deviceColor, seconds, m_stream);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA kernel: ") + cudaGetErrorString(status));

    float4 color{};
    status = cudaMemcpyAsync(&color, m_deviceColor, sizeof(color), cudaMemcpyDeviceToHost, m_stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(m_stream);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA color readback: ") + cudaGetErrorString(status));
    return color;
}
