#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>

class InteropSemaphore {
public:
    explicit InteropSemaphore(VkDevice device);
    ~InteropSemaphore();
    InteropSemaphore(const InteropSemaphore&) = delete;
    InteropSemaphore& operator=(const InteropSemaphore&) = delete;
    VkSemaphore vulkan() const noexcept { return m_vulkan; }
    cudaExternalSemaphore_t cuda() const noexcept { return m_cuda; }
private:
    VkDevice m_device;
    VkSemaphore m_vulkan = VK_NULL_HANDLE;
    cudaExternalSemaphore_t m_cuda = nullptr;
};
