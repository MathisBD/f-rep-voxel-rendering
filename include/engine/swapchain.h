#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "VkBootstrap.h"


class Swapchain
{
public:
    void Init(
        const vkb::Device& vkbDevice,
        const VkExtent2D& windowExtent);
    void Cleanup(const VkDevice& device);
    
    // request an image to render to.
    // this is blocking if there is no image ready.
    uint32_t RequestNewImage(
        const VkDevice& device,
        const VkSemaphore& presentSemaphore);
    
    const VkSwapchainKHR& GetSwapchain() { return m_swapchain; };
    const VkRenderPass& GetRenderPass() { return m_renderPass; };
    const std::vector<VkFramebuffer>& GetFramebuffers() { return m_framebuffers; };
private:
    VkSwapchainKHR m_swapchain;
    VkFormat m_imageFormat;
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;

    VkRenderPass m_renderPass;
    std::vector<VkFramebuffer> m_framebuffers;
};