#include "App.hpp"

#include "core/Config.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

App::App()
{
    if (!SDL_Init(SDL_INIT_VIDEO))
        throw std::runtime_error(std::string("SDL_Init: ") + SDL_GetError());

    try {
        m_window = SDL_CreateWindow(Config::WindowTitle, Config::WindowWidth,
                                    Config::WindowHeight, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
        if (!m_window)
            throw std::runtime_error(std::string("SDL_CreateWindow: ") + SDL_GetError());

        m_vulkanContext = std::make_unique<VulkanContext>(m_window);
        m_cudaContext = std::make_unique<CudaContext>(m_vulkanContext->physicalDevice());
        m_renderer = std::make_unique<Renderer>(*m_vulkanContext, *m_cudaContext, m_window);
        std::cout << "SDL + Vulkan + CUDA ready. Close the window or press Escape.\n";
    } catch (...) {
        m_renderer.reset();
        m_cudaContext.reset();
        m_vulkanContext.reset();
        if (m_window) SDL_DestroyWindow(m_window);
        SDL_Quit();
        throw;
    }
}

App::~App()
{
    m_renderer.reset();
    m_cudaContext.reset();
    m_vulkanContext.reset();
    if (m_window) SDL_DestroyWindow(m_window);
    SDL_Quit();
}

void App::run(unsigned int maxFrames)
{
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    bool running = true;
    unsigned int frames = 0;

    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT ||
                (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE))
                running = false;
        }
        if (!running) break;

        const float seconds = std::chrono::duration<float>(Clock::now() - start).count();
        m_renderer->render(seconds);
        if (maxFrames && ++frames >= maxFrames) running = false;
    }
}
