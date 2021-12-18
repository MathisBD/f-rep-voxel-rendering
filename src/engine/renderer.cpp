#include "engine/renderer.h"
#include "vk_wrapper/vk_check.h"
#include <glm/glm.hpp>
#include "vk_wrapper/initializers.h"


void Renderer::Init(const vkw::Device* dev, VkSurfaceKHR surface, VkExtent2D windowExtent) 
{
    device = dev->logicalDevice;

    m_frameNumber = 0;
    m_windowExtent = windowExtent;

	graphicsQueueFamily = dev->queueFamilies.graphics;
    vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue);

    swapchain.Init(dev, surface, windowExtent);
    for (size_t i = 0; i < FRAME_OVERLAP; i++) {
        m_frames[i].Init(dev);
    }
}

void Renderer::Cleanup() 
{    
    for (size_t i = 0; i < FRAME_OVERLAP; i++) {
        m_frames[i].Cleanup();
    }
    swapchain.Cleanup();
}

Frame& Renderer::CurrentFrame() 
{
    return m_frames[m_frameNumber % FRAME_OVERLAP];    
}

void Renderer::Draw(const DrawInfo* info)
{
    // wait until the GPU has finished rendering the previous frame. Timeout of 1 second.
	const VkFence& fence = CurrentFrame().renderFinishedFence;
    VK_CHECK(vkWaitForFences(device, 1, &fence, true, 1000000000));
	VK_CHECK(vkResetFences(device, 1, &fence));

    // Request a new image
    uint32_t swapchainImgIdx = swapchain.AcquireNewImage(CurrentFrame().imageReadySem);

    // now that we are sure that the commands finished executing, 
    // we can safely reset the command buffer to begin recording again.
	const VkCommandBuffer& cmd = CurrentFrame().commandBuffer;
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
	BuildRenderCommand(cmd, info, swapchainImgIdx);
    SubmitRenderCommand(cmd);
    PresentImage(swapchainImgIdx);

    m_frameNumber++;
}

void Renderer::BuildRenderCommand(
    VkCommandBuffer cmd, 
    const DrawInfo* info,
    uint32_t swapchainImgIdx) 
{
    // Begin command buffer
    auto beginInfo = vkw::init::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
    
    // Begin renderpass
    VkRenderPassBeginInfo rpInfo = { };
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.pNext = nullptr;
    
    rpInfo.renderPass = swapchain.renderPass;
    rpInfo.renderArea.offset.x = 0;
    rpInfo.renderArea.offset.y = 0;
    rpInfo.renderArea.extent = m_windowExtent;
    rpInfo.framebuffer = swapchain.framebuffers[swapchainImgIdx];

    VkClearValue clearColor;
    float flash = glm::abs(glm::sin(m_frameNumber / 60.0f));
    clearColor.color = { { 0.0f, flash, 0.0f , 1.0f } };
    rpInfo.clearValueCount = 1;
    rpInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info->pipeline);
    for (size_t i = 0; i < info->descriptorSets.size(); i++) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            info->pipelineLayout, 
            0, info->descriptorSets.size(), info->descriptorSets.data(), 
            0, nullptr);
    }
    vkCmdDraw(cmd, 6, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));
}

void Renderer::SubmitRenderCommand(VkCommandBuffer cmd) 
{
    auto submit = vkw::init::SubmitInfo(&cmd);
	
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &(CurrentFrame().imageReadySem);

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &(CurrentFrame().renderFinishedSem);

	// submit command buffer to the queue and execute it.
	// renderFinishedFence will now block until the graphic commands finish execution
	VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submit, CurrentFrame().renderFinishedFence));
}

void Renderer::PresentImage(uint32_t swapchainImgIdx) 
{
    // this will put the image we just rendered into the visible window.
	// we want to wait on the renderSemaphore for that,
	// as it's necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR info = {};
	info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	info.pNext = nullptr;

	info.pSwapchains = &swapchain.swapchain;
	info.swapchainCount = 1;

	info.pWaitSemaphores = &CurrentFrame().renderFinishedSem;
	info.waitSemaphoreCount = 1;

	info.pImageIndices = &swapchainImgIdx;

	VK_CHECK(vkQueuePresentKHR(graphicsQueue, &info));    
}