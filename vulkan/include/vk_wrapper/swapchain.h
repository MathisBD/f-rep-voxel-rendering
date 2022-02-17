#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "vk_wrapper/device.h"



namespace vkw
{
    class Swapchain
    {
    public:
        VkDevice device;
        VkExtent2D windowExtent;

        VkSwapchainKHR swapchain;
        VkFormat imageFormat;
        std::vector<VkImage> images;
        std::vector<VkImageView> imageViews;
        std::vector<VkFramebuffer> framebuffers;

        void Init(const vkw::Device* dev, VkSurfaceKHR surface, VkExtent2D windowExtent);
        void CreateFramebuffers(VkRenderPass rp);
        void Cleanup();
        // Request an image to render to.
        // This is blocking if there is no image ready.
        uint32_t AcquireNewImage(VkSemaphore presentSemaphore);
    };
}