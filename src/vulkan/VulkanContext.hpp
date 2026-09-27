#pragma once

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>

#include <cstdint>

class VulkanContext
{
public:
    explicit VulkanContext(SDL_Window* window);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    [[nodiscard]]
    VkInstance instance() const noexcept
    {
        return m_instance;
    }

    [[nodiscard]]
    VkPhysicalDevice physicalDevice() const noexcept
    {
        return m_physicalDevice;
    }

    [[nodiscard]]
    VkDevice device() const noexcept
    {
        return m_device;
    }

    [[nodiscard]]
    VkSurfaceKHR surface() const noexcept
    {
        return m_surface;
    }

    [[nodiscard]]
    VkQueue graphicsQueue() const noexcept
    {
        return m_graphicsQueue;
    }

    [[nodiscard]]
    uint32_t graphicsQueueFamily() const noexcept
    {
        return m_graphicsQueueFamily;
    }

private:
    void createInstance();
    void createSurface(SDL_Window* window);

    void selectPhysicalDevice();
    void createDevice();
    void destroy() noexcept;

private:
    VkInstance m_instance = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;

    VkPhysicalDevice m_physicalDevice =
        VK_NULL_HANDLE;

    VkDevice m_device = VK_NULL_HANDLE;

    VkQueue m_graphicsQueue =
        VK_NULL_HANDLE;

    uint32_t m_graphicsQueueFamily =
        UINT32_MAX;
};
