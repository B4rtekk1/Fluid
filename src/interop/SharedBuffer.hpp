#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>

#include <cstddef>

class SharedBuffer
{
public:
    SharedBuffer(
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        VkDeviceSize size,
        VkBufferUsageFlags usage
    );

    ~SharedBuffer();

    SharedBuffer(const SharedBuffer&) = delete;
    SharedBuffer& operator=(
        const SharedBuffer&) = delete;

    [[nodiscard]]
    VkBuffer buffer() const noexcept
    {
        return m_buffer;
    }

    [[nodiscard]]
    void* cudaPtr() const noexcept
    {
        return m_cudaPtr;
    }

    template<typename T>
    T* cudaPtrAs() const noexcept
    {
        return static_cast<T*>(
            m_cudaPtr
        );
    }

private:
    void createVulkanBuffer();
    void exportToCuda();

private:
    VkPhysicalDevice m_physicalDevice =
        VK_NULL_HANDLE;

    VkDevice m_device =
        VK_NULL_HANDLE;

    VkBuffer m_buffer =
        VK_NULL_HANDLE;

    VkDeviceMemory m_memory =
        VK_NULL_HANDLE;

    VkDeviceSize m_size = 0;

    VkBufferUsageFlags m_usage = 0;

    cudaExternalMemory_t m_cudaMemory =
        nullptr;

    void* m_cudaPtr = nullptr;
};