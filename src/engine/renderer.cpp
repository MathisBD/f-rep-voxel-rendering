#include "engine/renderer.h"
#include "engine/vk_check.h"
#include <glm/glm.hpp>


void Renderer::Init(
    const vkb::Device& vkbDevice,
    const VkExtent2D& windowExtent) 
{
    m_frameNumber = 0;
    m_windowExtent = windowExtent;

    m_graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	m_graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    m_swapchain.Init(vkbDevice, windowExtent);
    m_frame.Init(vkbDevice.device, m_graphicsQueueFamily);
}

void Renderer::Cleanup(const VkDevice& device) 
{    
    m_frame.Cleanup(device);
    m_swapchain.Cleanup(device);
}

void Renderer::Draw(const VkDevice& device) 
{
    // wait until the GPU has finished rendering the previous frame. Timeout of 1 second.
	VK_CHECK(vkWaitForFences(device, 1, &m_frame.GetRenderFence(), true, 1000000000));
	VK_CHECK(vkResetFences(device, 1, &m_frame.GetRenderFence()));

    // Request a new image
    uint32_t swapchainImgIdx = m_swapchain.RequestNewImage(
        device, m_frame.GetPresentSemaphore());

    // now that we are sure that the commands finished executing, 
    // we can safely reset the command buffer to begin recording again.
	const VkCommandBuffer& cmd = m_frame.GetCommandBuffer();
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
	BuildRenderCommand(cmd, swapchainImgIdx);
    SubmitRenderCommand(cmd);
    PresentImage(swapchainImgIdx);

    m_frameNumber++;
}

void Renderer::BuildRenderCommand(const VkCommandBuffer& cmd, uint32_t swapchainImgIdx) 
{
    // Begin command buffer
    VkCommandBufferBeginInfo cmdInfo = { };
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdInfo.pNext = nullptr;
    
    cmdInfo.pInheritanceInfo = nullptr;
    cmdInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdInfo));
    
    // Begin renderpass
    VkRenderPassBeginInfo rpInfo = { };
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.pNext = nullptr;
    
    rpInfo.renderPass = m_swapchain.GetRenderPass();
    rpInfo.renderArea.offset.x = 0;
    rpInfo.renderArea.offset.y = 0;
    rpInfo.renderArea.extent = m_windowExtent;
    rpInfo.framebuffer = m_swapchain.GetFramebuffers()[swapchainImgIdx];

    VkClearValue clearColor;
    float flash = glm::abs(glm::sin(m_frameNumber / 60.0f));
    clearColor.color = { { 0.0f, flash, 0.0f , 1.0f } };
    rpInfo.clearValueCount = 1;
    rpInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    // TODO : render some geometry

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(m_frame.GetCommandBuffer()));
}

void Renderer::SubmitRenderCommand(const VkCommandBuffer& cmd) 
{
    VkSubmitInfo submit = {};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext = nullptr;

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &m_frame.GetPresentSemaphore();

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &m_frame.GetRenderSemaphore();

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;

	//submit command buffer to the queue and execute it.
	// _renderFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(m_graphicsQueue, 1, &submit, m_frame.GetRenderFence()));
}

void Renderer::PresentImage(uint32_t swapchainImgIdx) 
{
    // this will put the image we just rendered into the visible window.
	// we want to wait on the renderSemaphore for that,
	// as it's necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR info = {};
	info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	info.pNext = nullptr;

	info.pSwapchains = &m_swapchain.GetSwapchain();
	info.swapchainCount = 1;

	info.pWaitSemaphores = &m_frame.GetRenderSemaphore();
	info.waitSemaphoreCount = 1;

	info.pImageIndices = &swapchainImgIdx;

	VK_CHECK(vkQueuePresentKHR(m_graphicsQueue, &info));    
}