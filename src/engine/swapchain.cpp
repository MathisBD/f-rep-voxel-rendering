#include "engine/swapchain.h"
#include "VkBootstrap.h"


void Swapchain::Init(
    const VkPhysicalDevice& gpu, 
    const VkDevice& device, 
    const VkSurfaceKHR& surface) 
{
    vkb::SwapchainBuilder builder(gpu, device, surface);  
    vkb::Swapchain vkbSwapchain = builder
        .use_default_format_selection()
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .build()
        .value();

    m_swapchain = vkbSwapchain.swapchain;
    m_imageFormat = vkbSwapchain.image_format;
    m_images = vkbSwapchain.get_images().value();
    m_imageViews = vkbSwapchain.get_image_views().value();
}

void Swapchain::Cleanup(const VkDevice& device) 
{
    vkDestroySwapchainKHR(device, m_swapchain, nullptr);
    for (const auto& view : m_imageViews) {
        vkDestroyImageView(device, view, nullptr);
	}    
}