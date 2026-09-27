#include "CudaContext.hpp"
#include "Simulation.cuh"

#include <cstring>
#include <stdexcept>
#include <string>

namespace
{
    void checkCudaAvailable()
    {
        int deviceCount = 0;

        const cudaError_t result =
            cudaGetDeviceCount(&deviceCount);

        if (result == cudaErrorNoDevice)
        {
            throw std::runtime_error(
                "CUDA is not available: "
                "no CUDA-capable NVIDIA GPU was detected."
            );
        }

        if (result == cudaErrorInsufficientDriver)
        {
            throw std::runtime_error(
                "CUDA is not available: "
                "the NVIDIA driver is missing or too old."
            );
        }

        if (result == cudaErrorInitializationError)
        {
            throw std::runtime_error(
                "CUDA initialization failed."
            );
        }

        if (result != cudaSuccess)
        {
            throw std::runtime_error(
                std::string(
                    "CUDA initialization failed: "
                ) +
                cudaGetErrorString(result)
            );
        }

        if (deviceCount == 0)
        {
            throw std::runtime_error(
                "CUDA is not available: "
                "no CUDA-capable GPU was detected."
            );
        }
    }
}

CudaContext::CudaContext(
    VkPhysicalDevice physicalDevice)
{
    checkCudaAvailable();

    m_deviceIndex =
        findMatchingDevice(physicalDevice);

    if (m_deviceIndex < 0)
    {
        throw std::runtime_error(
            "CUDA is available, but the Vulkan GPU "
            "does not match any CUDA device."
        );
    }

    cudaError_t result =
        cudaSetDevice(m_deviceIndex);

    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("cudaSetDevice failed: ") +
            cudaGetErrorString(result)
        );
    }

    result = cudaStreamCreateWithFlags(
        &m_stream,
        cudaStreamNonBlocking
    );

    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(
                "cudaStreamCreateWithFlags failed: "
            ) +
            cudaGetErrorString(result)
        );
    }

    result = cudaMalloc(&m_deviceColor, sizeof(float4));
    if (result != cudaSuccess)
    {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
        throw std::runtime_error(
            std::string("cudaMalloc failed: ") +
            cudaGetErrorString(result)
        );
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

    const cudaError_t countResult =
        cudaGetDeviceCount(&cudaDeviceCount);

    if (countResult != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(
                "cudaGetDeviceCount failed: "
            ) +
            cudaGetErrorString(countResult)
        );
    }

    for (int i = 0;
         i < cudaDeviceCount;
         ++i)
    {
        cudaDeviceProp cudaProperties{};

        const cudaError_t result =
            cudaGetDeviceProperties(
            &cudaProperties,
            i
        );

        if (result != cudaSuccess)
            continue;

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
