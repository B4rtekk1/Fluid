#include "InteropSemaphore.hpp"

#include <windows.h>
#include <stdexcept>
#include <string>

InteropSemaphore::InteropSemaphore(VkDevice device) : m_device(device) {
    VkExportSemaphoreCreateInfo exportInfo{
        .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
    };
    VkSemaphoreCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &exportInfo
    };
    if (vkCreateSemaphore(device, &info, nullptr, &m_vulkan) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSemaphore for CUDA interop failed");
    try {
        auto getHandle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
        if (!getHandle) throw std::runtime_error("vkGetSemaphoreWin32HandleKHR unavailable");
        VkSemaphoreGetWin32HandleInfoKHR handleInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
            .semaphore = m_vulkan,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
        };
        HANDLE handle = nullptr;
        if (getHandle(device, &handleInfo, &handle) != VK_SUCCESS)
            throw std::runtime_error("Exporting Vulkan semaphore handle failed");
        cudaExternalSemaphoreHandleDesc descriptor{};
        descriptor.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        descriptor.handle.win32.handle = handle;
        const cudaError_t result = cudaImportExternalSemaphore(&m_cuda, &descriptor);
        CloseHandle(handle);
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("cudaImportExternalSemaphore: ") + cudaGetErrorString(result));
    } catch (...) {
        vkDestroySemaphore(device, m_vulkan, nullptr);
        throw;
    }
}

InteropSemaphore::~InteropSemaphore() {
    if (m_cuda) cudaDestroyExternalSemaphore(m_cuda);
    if (m_vulkan) vkDestroySemaphore(m_device, m_vulkan, nullptr);
}
