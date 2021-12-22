#include "vk_wrapper/swapchain.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/device.h"
#include "VkBootstrap.h"


void vkw::Swapchain::Init(
	const vkw::Device* dev, 
	VkSurfaceKHR surface, 
	VkExtent2D windowExtent) 
{
	device = dev->logicalDevice;
	this->windowExtent = windowExtent;

    // Swapchain and images
    vkb::SwapchainBuilder builder(
		dev->physicalDevice, 
		dev->logicalDevice, 
		surface);  
    vkb::Swapchain vkbSwapchain = builder
        .use_default_format_selection()
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(windowExtent.width, windowExtent.height)
        .build()
        .value();

    swapchain = vkbSwapchain.swapchain;
    imageFormat = vkbSwapchain.image_format;
    images = vkbSwapchain.get_images().value();
    imageViews = vkbSwapchain.get_image_views().value();
}

void vkw::Swapchain::CreateFramebuffers(VkRenderPass rp)
{
    // Framebuffers
	framebuffers = std::vector<VkFramebuffer>(images.size());
	for (size_t i = 0; i < images.size(); i++) {
        VkFramebufferCreateInfo fbInfo = {};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.pNext = nullptr;

        fbInfo.renderPass = rp;
        fbInfo.attachmentCount = 1;
        fbInfo.width = windowExtent.width;
        fbInfo.height = windowExtent.height;
        fbInfo.layers = 1;

		fbInfo.pAttachments = &imageViews[i];
		VK_CHECK(vkCreateFramebuffer(device, &fbInfo, nullptr, &framebuffers[i]));
	}
}

void vkw::Swapchain::Cleanup() 
{
	vkDestroySwapchainKHR(device, swapchain, nullptr);
    for (size_t i = 0; i < images.size(); i++) {
		vkDestroyFramebuffer(device, framebuffers[i], nullptr);
        vkDestroyImageView(device, imageViews[i], nullptr);
	}    
}

uint32_t vkw::Swapchain::AcquireNewImage(VkSemaphore sem) 
{
	uint32_t imgIdx;
	// 1 second timeout
	VK_CHECK(vkAcquireNextImageKHR(
		device, swapchain, 1000000000, sem, nullptr, &imgIdx));
	return imgIdx;	
}
