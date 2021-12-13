#pragma once
#include <vulkan/vulkan.h>
#include <vector>


class Swapchain
{
public:
    void Init(
        const VkPhysicalDevice& gpu, 
        const VkDevice& device, 
        const VkSurfaceKHR& surface);
    void Cleanup(const VkDevice& device);
private:
    VkSwapchainKHR m_swapchain;
    VkFormat m_imageFormat;
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;
};