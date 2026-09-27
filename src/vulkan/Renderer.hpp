#pragma once

#include "VulkanContext.hpp"
#include "cuda/CudaContext.hpp"
#include "interop/SharedBuffer.hpp"
#include "interop/InteropSemaphore.hpp"

#include <SDL3/SDL.h>
#include <vector>
#include <array>
#include <memory>

class Renderer {
public:
    Renderer(VulkanContext& context, CudaContext& cuda, SDL_Window* window);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void render(float seconds);

private:
    void createSwapchain();
    void destroySwapchain() noexcept;
    void recreateSwapchain();
    struct Frame {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        std::unique_ptr<SharedBuffer> pixels;
        std::unique_ptr<InteropSemaphore> cudaReady;
    };

    VulkanContext& m_context;
    CudaContext& m_cuda;
    SDL_Window* m_window;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> m_images;
    std::vector<VkSemaphore> m_renderFinished;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::array<Frame, 2> m_frames;
    size_t m_frameIndex = 0;
    VkExtent2D m_extent{};
    VkFormat m_format = VK_FORMAT_UNDEFINED;
};
