#pragma once

#include <SDL3/SDL.h>
#include <memory>

#include "cuda/CudaContext.hpp"
#include "vulkan/Renderer.hpp"
#include "vulkan/VulkanContext.hpp"

class App {
public:
    App();
    ~App();

    App(const App&) = delete;
    App& operator=(const App&) = delete;

    void run(unsigned int maxFrames = 0);

private:
    SDL_Window* m_window = nullptr;
    std::unique_ptr<VulkanContext> m_vulkanContext;
    std::unique_ptr<CudaContext> m_cudaContext;
    std::unique_ptr<Renderer> m_renderer;
};
