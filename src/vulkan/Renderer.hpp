#pragma once

#include "VulkanContext.hpp"

#include <SDL3/SDL.h>
#include <vector>

class Renderer {
public:
    Renderer(VulkanContext& context, SDL_Window* window);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void render(float red, float green, float blue);

private:
    void createSwapchain();
    void destroySwapchain() noexcept;
    void recreateSwapchain();

    VulkanContext& m_context;
    SDL_Window* m_window;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> m_images;
    std::vector<VkSemaphore> m_renderFinished;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkSemaphore m_imageAvailable = VK_NULL_HANDLE;
};
