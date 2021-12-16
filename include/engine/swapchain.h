#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "vk_wrapper/device.h"



class Swapchain
{
public:
    VkDevice device;

    VkSwapchainKHR swapchain;
    VkFormat imageFormat;
    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;

    VkRenderPass renderPass;
    std::vector<VkFramebuffer> framebuffers;

    void Init(const vkw::Device* dev, VkSurfaceKHR surface, VkExtent2D windowExtent);
    void Cleanup();
    // request an image to render to.
    // this is blocking if there is no image ready.
    uint32_t AcquireNewImage(VkSemaphore presentSemaphore);
};