#include "SharedBuffer.hpp"

#include <windows.h>
#include <stdexcept>
#include <string>

SharedBuffer::SharedBuffer(VkPhysicalDevice physicalDevice, VkDevice device,
                           VkDeviceSize size, VkBufferUsageFlags usage)
    : m_physicalDevice(physicalDevice), m_device(device), m_size(size), m_usage(usage)
{
    try { createVulkanBuffer(); exportToCuda(); }
    catch (...) {
        if (m_cudaPtr) cudaFree(m_cudaPtr);
        if (m_cudaMemory) cudaDestroyExternalMemory(m_cudaMemory);
        if (m_buffer) vkDestroyBuffer(m_device, m_buffer, nullptr);
        if (m_memory) vkFreeMemory(m_device, m_memory, nullptr);
        throw;
    }
}

SharedBuffer::~SharedBuffer()
{
    if (m_cudaPtr) cudaFree(m_cudaPtr);
    if (m_cudaMemory) cudaDestroyExternalMemory(m_cudaMemory);
    if (m_buffer) vkDestroyBuffer(m_device, m_buffer, nullptr);
    if (m_memory) vkFreeMemory(m_device, m_memory, nullptr);
}

void SharedBuffer::createVulkanBuffer()
{
    VkExternalMemoryBufferCreateInfo external{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT};
    VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .pNext = &external, .size = m_size,
                            .usage = m_usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(m_device, &info, nullptr, &m_buffer) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer for CUDA interop failed");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(m_device, m_buffer, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &properties);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = i;
            break;
        }
    if (type == UINT32_MAX) throw std::runtime_error("No exportable device-local memory type");
    VkExportMemoryAllocateInfo exportInfo{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT};
    VkMemoryDedicatedAllocateInfo dedicated{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &exportInfo, .buffer = m_buffer};
    VkMemoryAllocateInfo allocate{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated, .allocationSize = requirements.size, .memoryTypeIndex = type};
    if (vkAllocateMemory(m_device, &allocate, nullptr, &m_memory) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory for CUDA interop failed");
    if (vkBindBufferMemory(m_device, m_buffer, m_memory, 0) != VK_SUCCESS)
        throw std::runtime_error("vkBindBufferMemory failed");
    m_size = requirements.size;
}

void SharedBuffer::exportToCuda()
{
    auto getHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(m_device, "vkGetMemoryWin32HandleKHR"));
    if (!getHandle) throw std::runtime_error("vkGetMemoryWin32HandleKHR unavailable");
    VkMemoryGetWin32HandleInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
        .memory = m_memory, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT};
    HANDLE handle = nullptr;
    if (getHandle(m_device, &info, &handle) != VK_SUCCESS)
        throw std::runtime_error("Exporting Vulkan buffer handle failed");
    cudaExternalMemoryHandleDesc descriptor{};
    descriptor.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    descriptor.handle.win32.handle = handle;
    descriptor.size = m_size;
    descriptor.flags = cudaExternalMemoryDedicated;
    const cudaError_t imported = cudaImportExternalMemory(&m_cudaMemory, &descriptor);
    CloseHandle(handle);
    if (imported != cudaSuccess)
        throw std::runtime_error(std::string("cudaImportExternalMemory: ") + cudaGetErrorString(imported));
    cudaExternalMemoryBufferDesc mapping{};
    mapping.size = m_size;
    const cudaError_t mapped = cudaExternalMemoryGetMappedBuffer(&m_cudaPtr, m_cudaMemory, &mapping);
    if (mapped != cudaSuccess)
        throw std::runtime_error(std::string("cudaExternalMemoryGetMappedBuffer: ") + cudaGetErrorString(mapped));
}
