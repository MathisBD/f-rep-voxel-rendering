#include "engine/frame.h"
#include "engine/vk_check.h"
#include "engine/vk_init.h"


void Frame::Init(
        const VkDevice& device,
        uint32_t graphicsQueueFamily) 
{
    // Command pool
    auto poolInfo = vkinit::CommandPoolCreateInfo(
        graphicsQueueFamily, 
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &m_commandPool));

    // Command buffer
    auto cmdInfo = vkinit::CommandBufferAllocateInfo(m_commandPool);
    VK_CHECK(vkAllocateCommandBuffers(device, &cmdInfo, &m_commandBuffer));

    // Render fence
    auto fenceInfo = vkinit::FenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &m_renderFence));

    // Semaphores
    auto semInfo = vkinit::SemaphoreCreateInfo();
    VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &m_presentSemaphore));
    VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &m_renderSemaphore));
}

void Frame::Cleanup(const VkDevice& device) 
{
    // make sure the GPU has stopped rendering
	vkWaitForFences(device, 1, &m_renderFence, true, 1000000000);
    // only now destroy the synchronization variables.
    vkDestroyFence(device, m_renderFence, nullptr);
    vkDestroySemaphore(device, m_presentSemaphore, nullptr);
    vkDestroySemaphore(device, m_renderSemaphore, nullptr);

    vkDestroyCommandPool(device, m_commandPool, nullptr);    
}
