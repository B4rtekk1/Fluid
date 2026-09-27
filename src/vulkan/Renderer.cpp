#include "Renderer.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

Renderer::Renderer(VulkanContext& context, SDL_Window* window)
    : m_context(context), m_window(window)
{
    try {
        VkCommandPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                          .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                          .queueFamilyIndex = context.graphicsQueueFamily()};
        if (vkCreateCommandPool(context.device(), &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS)
            throw std::runtime_error("vkCreateCommandPool failed");
        VkCommandBufferAllocateInfo commandInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                 .commandPool = m_commandPool,
                                                 .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                 .commandBufferCount = 1};
        if (vkAllocateCommandBuffers(context.device(), &commandInfo, &m_commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("vkAllocateCommandBuffers failed");
        VkSemaphoreCreateInfo semaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (vkCreateSemaphore(context.device(), &semaphoreInfo, nullptr, &m_imageAvailable) != VK_SUCCESS)
            throw std::runtime_error("vkCreateSemaphore failed");
        createSwapchain();
    } catch (...) {
        destroySwapchain();
        if (m_imageAvailable) vkDestroySemaphore(context.device(), m_imageAvailable, nullptr);
        if (m_commandPool) vkDestroyCommandPool(context.device(), m_commandPool, nullptr);
        throw;
    }
}

Renderer::~Renderer()
{
    vkDeviceWaitIdle(m_context.device());
    destroySwapchain();
    if (m_imageAvailable) vkDestroySemaphore(m_context.device(), m_imageAvailable, nullptr);
    if (m_commandPool) vkDestroyCommandPool(m_context.device(), m_commandPool, nullptr);
}

void Renderer::destroySwapchain() noexcept
{
    for (VkSemaphore semaphore : m_renderFinished)
        vkDestroySemaphore(m_context.device(), semaphore, nullptr);
    m_renderFinished.clear();
    m_images.clear();
    if (m_swapchain) vkDestroySwapchainKHR(m_context.device(), m_swapchain, nullptr);
    m_swapchain = VK_NULL_HANDLE;
}

void Renderer::createSwapchain()
{
    VkSurfaceCapabilitiesKHR capabilities{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_context.physicalDevice(), m_context.surface(),
                                                   &capabilities) != VK_SUCCESS)
        throw std::runtime_error("vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed");

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_context.physicalDevice(), m_context.surface(), &formatCount, nullptr);
    if (formatCount == 0) throw std::runtime_error("No Vulkan surface formats");
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_context.physicalDevice(), m_context.surface(), &formatCount, formats.data());
    VkSurfaceFormatKHR format = formats.front();
    for (const auto& candidate : formats)
        if ((candidate.format == VK_FORMAT_B8G8R8A8_UNORM || candidate.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            candidate.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            format = candidate;
            break;
        }

    VkExtent2D extent = capabilities.currentExtent;
    if (extent.width == UINT32_MAX) {
        int width = 0, height = 0;
        SDL_GetWindowSizeInPixels(m_window, &width, &height);
        extent.width = std::clamp(static_cast<uint32_t>(std::max(width, 1)),
                                  capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(static_cast<uint32_t>(std::max(height, 1)),
                                   capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    VkSwapchainCreateInfoKHR info{.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                                   .surface = m_context.surface(),
                                   .minImageCount = std::min(capabilities.minImageCount + 1,
                                       capabilities.maxImageCount ? capabilities.maxImageCount : UINT32_MAX),
                                   .imageFormat = format.format,
                                   .imageColorSpace = format.colorSpace,
                                   .imageExtent = extent,
                                   .imageArrayLayers = 1,
                                   .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                   .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                   .preTransform = capabilities.currentTransform,
                                   .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                                   .presentMode = VK_PRESENT_MODE_FIFO_KHR,
                                   .clipped = VK_TRUE};
    if (!(capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) {
        for (VkCompositeAlphaFlagBitsKHR flag : {VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                                                 VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                                                 VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
            if (capabilities.supportedCompositeAlpha & flag) { info.compositeAlpha = flag; break; }
    }
    if (vkCreateSwapchainKHR(m_context.device(), &info, nullptr, &m_swapchain) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSwapchainKHR failed");

    uint32_t imageCount = 0;
    if (vkGetSwapchainImagesKHR(m_context.device(), m_swapchain, &imageCount, nullptr) != VK_SUCCESS || !imageCount)
        throw std::runtime_error("vkGetSwapchainImagesKHR failed");
    m_images.resize(imageCount);
    if (vkGetSwapchainImagesKHR(m_context.device(), m_swapchain, &imageCount, m_images.data()) != VK_SUCCESS)
        throw std::runtime_error("vkGetSwapchainImagesKHR failed");
    VkSemaphoreCreateInfo semaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        if (vkCreateSemaphore(m_context.device(), &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS)
            throw std::runtime_error("vkCreateSemaphore failed");
        m_renderFinished.push_back(semaphore);
    }
}

void Renderer::recreateSwapchain()
{
    vkDeviceWaitIdle(m_context.device());
    destroySwapchain();
    createSwapchain();
}

void Renderer::render(float red, float green, float blue)
{
    int width = 0, height = 0;
    SDL_GetWindowSizeInPixels(m_window, &width, &height);
    if (width == 0 || height == 0) {
        SDL_Delay(16);
        return;
    }

    uint32_t imageIndex = 0;
    const VkResult acquired = vkAcquireNextImageKHR(m_context.device(), m_swapchain, UINT64_MAX,
                                                     m_imageAvailable, VK_NULL_HANDLE, &imageIndex);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("vkAcquireNextImageKHR failed");

    if (vkResetCommandBuffer(m_commandBuffer, 0) != VK_SUCCESS)
        throw std::runtime_error("vkResetCommandBuffer failed");
    VkCommandBufferBeginInfo beginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (vkBeginCommandBuffer(m_commandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("vkBeginCommandBuffer failed");

    VkImageMemoryBarrier toTransfer{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                     .srcAccessMask = 0,
                                     .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                     .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                     .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                     .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                     .image = m_images[imageIndex],
                                     .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransfer);
    VkClearColorValue clearColor{{red, green, blue, 1.0f}};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(m_commandBuffer, m_images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &clearColor, 1, &range);

    VkImageMemoryBarrier toPresent = toTransfer;
    toPresent.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toPresent.dstAccessMask = 0;
    toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &toPresent);
    if (vkEndCommandBuffer(m_commandBuffer) != VK_SUCCESS)
        throw std::runtime_error("vkEndCommandBuffer failed");

    constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                         .waitSemaphoreCount = 1,
                         .pWaitSemaphores = &m_imageAvailable,
                         .pWaitDstStageMask = &waitStage,
                         .commandBufferCount = 1,
                         .pCommandBuffers = &m_commandBuffer,
                         .signalSemaphoreCount = 1,
                         .pSignalSemaphores = &m_renderFinished[imageIndex]};
    if (vkQueueSubmit(m_context.graphicsQueue(), 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS)
        throw std::runtime_error("vkQueueSubmit failed");
    VkPresentInfoKHR present{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                             .waitSemaphoreCount = 1,
                             .pWaitSemaphores = &m_renderFinished[imageIndex],
                             .swapchainCount = 1,
                             .pSwapchains = &m_swapchain,
                             .pImageIndices = &imageIndex};
    const VkResult result = vkQueuePresentKHR(m_context.graphicsQueue(), &present);
    if (vkQueueWaitIdle(m_context.graphicsQueue()) != VK_SUCCESS)
        throw std::runtime_error("vkQueueWaitIdle failed");
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || acquired == VK_SUBOPTIMAL_KHR)
        recreateSwapchain();
    else if (result != VK_SUCCESS)
        throw std::runtime_error("vkQueuePresentKHR failed");
}
