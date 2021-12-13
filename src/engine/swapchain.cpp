#include "engine/swapchain.h"
#include "VkBootstrap.h"
#include "engine/vk_check.h"


void Swapchain::Init(
	const vkb::Device& vkbDevice,
    const VkExtent2D& windowExtent) 
{
    // Swapchain and images
    vkb::SwapchainBuilder builder(
		vkbDevice.physical_device, 
		vkbDevice.device, 
		vkbDevice.surface);  
    vkb::Swapchain vkbSwapchain = builder
        .use_default_format_selection()
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(windowExtent.width, windowExtent.height)
        .build()
        .value();

    m_swapchain = vkbSwapchain.swapchain;
    m_imageFormat = vkbSwapchain.image_format;
    m_images = vkbSwapchain.get_images().value();
    m_imageViews = vkbSwapchain.get_image_views().value();

    // Renderpass
	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = m_imageFormat;
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
	VK_CHECK(vkCreateRenderPass(vkbDevice.device, &renderPassInfo, nullptr, &m_renderPass));

    // Framebuffers
	m_framebuffers = std::vector<VkFramebuffer>(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
        VkFramebufferCreateInfo fbInfo = {};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.pNext = nullptr;

        fbInfo.renderPass = m_renderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.width = windowExtent.width;
        fbInfo.height = windowExtent.height;
        fbInfo.layers = 1;

		fbInfo.pAttachments = &m_imageViews[i];
		VK_CHECK(vkCreateFramebuffer(vkbDevice.device, &fbInfo, nullptr, &m_framebuffers[i]));
	}
}

void Swapchain::Cleanup(const VkDevice& device) 
{
	vkDestroyRenderPass(device, m_renderPass, nullptr);
    vkDestroySwapchainKHR(device, m_swapchain, nullptr);

    for (size_t i = 0; i < m_images.size(); i++) {
		vkDestroyFramebuffer(device, m_framebuffers[i], nullptr);
        vkDestroyImageView(device, m_imageViews[i], nullptr);
	}    
}

uint32_t Swapchain::RequestNewImage(
	const VkDevice& device,
	const VkSemaphore& presentSemaphore) 
{
	uint32_t imgIdx;
	VK_CHECK(vkAcquireNextImageKHR(
		device, 
		m_swapchain, 
		1000000000, // 1 second timeout
		presentSemaphore, 
		nullptr, 
		&imgIdx));
	return imgIdx;	
}
