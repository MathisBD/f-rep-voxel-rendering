#include "engine/swapchain.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/device.h"
#include "VkBootstrap.h"


void Swapchain::Init(
	const vkw::Device* dev, 
	VkSurfaceKHR surface, 
	VkExtent2D windowExtent) 
{
	device = dev->logicalDevice;

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

    // Renderpass
	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = imageFormat;
	// 1 sample, we won't be doing MSAA
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	// we clear when this attachment is loaded
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	// we keep the attachment stored when the renderpass ends
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	// we don't care about stencil
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // we don't know or care about the starting layout of the attachment
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    //after the renderpass ends, the image has to be on a layout ready for display
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
	// attachment number will index into the pAttachments array in the parent renderpass
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

    VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

	// connect the color attachment to the info
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	// connect the subpass to the info
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	VK_CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));

    // Framebuffers
	framebuffers = std::vector<VkFramebuffer>(images.size());
	for (size_t i = 0; i < images.size(); i++) {
        VkFramebufferCreateInfo fbInfo = {};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.pNext = nullptr;

        fbInfo.renderPass = renderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.width = windowExtent.width;
        fbInfo.height = windowExtent.height;
        fbInfo.layers = 1;

		fbInfo.pAttachments = &imageViews[i];
		VK_CHECK(vkCreateFramebuffer(device, &fbInfo, nullptr, &framebuffers[i]));
	}
}

void Swapchain::Cleanup() 
{
	vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);

    for (size_t i = 0; i < images.size(); i++) {
		vkDestroyFramebuffer(device, framebuffers[i], nullptr);
        vkDestroyImageView(device, imageViews[i], nullptr);
	}    
}

uint32_t Swapchain::AcquireNewImage(VkSemaphore sem) 
{
	uint32_t imgIdx;
	// 1 second timeout
	VK_CHECK(vkAcquireNextImageKHR(
		device, swapchain, 1000000000, sem, nullptr, &imgIdx));
	return imgIdx;	
}
