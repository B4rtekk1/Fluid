#include "VulkanContext.hpp"

#include <SDL3/SDL_vulkan.h>

#include <array>
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

VulkanContext::VulkanContext(SDL_Window* window)
{
    try {
        createInstance();
        createSurface(window);
        selectPhysicalDevice();
        createDevice();
    } catch (...) {
        destroy();
        throw;
    }
}

VulkanContext::~VulkanContext()
{
    destroy();
}

void VulkanContext::destroy() noexcept
{
    if (m_device)
    {
        vkDeviceWaitIdle(m_device);
        vkDestroyDevice(m_device, nullptr);
    }

    if (m_surface)
    {
        SDL_Vulkan_DestroySurface(
            m_instance,
            m_surface,
            nullptr
        );
    }

    if (m_instance)
    {
        vkDestroyInstance(
            m_instance,
            nullptr
        );
    }
}

void VulkanContext::createInstance()
{
    Uint32 sdlExtensionCount = 0;

    const char* const* sdlExtensions =
        SDL_Vulkan_GetInstanceExtensions(
            &sdlExtensionCount
        );

    if (!sdlExtensions)
    {
        throw std::runtime_error(
            "SDL_Vulkan_GetInstanceExtensions failed"
        );
    }

    std::vector<const char*> extensions(
        sdlExtensions,
        sdlExtensions + sdlExtensionCount
    );

    extensions.push_back(
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME
    );

    extensions.push_back(
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME
    );

    VkApplicationInfo appInfo{
        .sType =
            VK_STRUCTURE_TYPE_APPLICATION_INFO,

        .pApplicationName =
            "GPU PBF Fluid Simulator",

        .applicationVersion =
            VK_MAKE_VERSION(0, 1, 0),

        .pEngineName =
            "FluidPBF",

        .engineVersion =
            VK_MAKE_VERSION(0, 1, 0),

        .apiVersion =
            VK_API_VERSION_1_3
    };

    VkInstanceCreateInfo createInfo{
        .sType =
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,

        .pApplicationInfo =
            &appInfo,

        .enabledExtensionCount =
            static_cast<uint32_t>(
                extensions.size()
            ),

        .ppEnabledExtensionNames =
            extensions.data()
    };

    if (vkCreateInstance(
            &createInfo,
            nullptr,
            &m_instance) != VK_SUCCESS)
    {
        throw std::runtime_error(
            "vkCreateInstance failed"
        );
    }
}

void VulkanContext::createSurface(
    SDL_Window* window)
{
    if (!SDL_Vulkan_CreateSurface(
            window,
            m_instance,
            nullptr,
            &m_surface))
    {
        throw std::runtime_error(
            std::string(
                "SDL_Vulkan_CreateSurface failed: "
            ) +
            SDL_GetError()
        );
    }
}

void VulkanContext::selectPhysicalDevice()
{
    int cudaCount = 0;
    const cudaError_t cudaStatus = cudaGetDeviceCount(&cudaCount);
    if (cudaStatus != cudaSuccess || cudaCount == 0)
        throw std::runtime_error(std::string("No CUDA device: ") + cudaGetErrorString(cudaStatus));

    uint32_t deviceCount = 0;
    if (vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr) != VK_SUCCESS || deviceCount == 0)
        throw std::runtime_error("No Vulkan physical device");
    std::vector<VkPhysicalDevice> devices(deviceCount);
    if (vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data()) != VK_SUCCESS)
        throw std::runtime_error("vkEnumeratePhysicalDevices failed");

    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceIDProperties id{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                               .pNext = &id};
        vkGetPhysicalDeviceProperties2(device, &properties);
        bool cudaMatch = false;
        for (int i = 0; i < cudaCount; ++i) {
            cudaDeviceProp cudaProperties{};
            if (cudaGetDeviceProperties(&cudaProperties, i) == cudaSuccess &&
                std::memcmp(cudaProperties.uuid.bytes, id.deviceUUID, VK_UUID_SIZE) == 0) {
                cudaMatch = true;
                break;
            }
        }
        if (!cudaMatch) continue;

        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());
        bool hasSwapchain = false, hasMemory = false, hasMemoryWin32 = false;
        bool hasSemaphore = false, hasSemaphoreWin32 = false;
        for (const auto& extension : extensions)
        {
            hasSwapchain |= std::strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
            hasMemory |= std::strcmp(extension.extensionName, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) == 0;
            hasMemoryWin32 |= std::strcmp(extension.extensionName, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) == 0;
            hasSemaphore |= std::strcmp(extension.extensionName, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME) == 0;
            hasSemaphoreWin32 |= std::strcmp(extension.extensionName, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME) == 0;
        }
        if (!(hasSwapchain && hasMemory && hasMemoryWin32 && hasSemaphore && hasSemaphoreWin32)) continue;

        VkSurfaceCapabilitiesKHR capabilities{};
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface, &capabilities) != VK_SUCCESS ||
            !(capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) continue;

        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());
        for (uint32_t family = 0; family < familyCount; ++family) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, family, m_surface, &present);
            if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                m_physicalDevice = device;
                m_graphicsQueueFamily = family;
                return;
            }
        }
    }
    throw std::runtime_error("No CUDA-compatible Vulkan GPU can present to this SDL window");
}

void VulkanContext::createDevice()
{
    constexpr float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                      .queueFamilyIndex = m_graphicsQueueFamily,
                                      .queueCount = 1,
                                      .pQueuePriorities = &priority};
    constexpr const char* extensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
    VkDeviceCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                  .queueCreateInfoCount = 1,
                                  .pQueueCreateInfos = &queueInfo,
                                  .enabledExtensionCount = static_cast<uint32_t>(std::size(extensions)),
                                  .ppEnabledExtensionNames = extensions};
    if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed");
    vkGetDeviceQueue(m_device, m_graphicsQueueFamily, 0, &m_graphicsQueue);
}
