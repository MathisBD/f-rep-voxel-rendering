#include "engine/frame.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/initializers.h"


void Frame::Init(const vkw::Device* dev) 
{
    device = dev->logicalDevice;

    // Command pool
    auto poolInfo = vkw::init::CommandPoolCreateInfo(
        dev->queueFamilies.graphics, 
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));

    // Command buffer
    auto cmdInfo = vkw::init::CommandBufferAllocateInfo(commandPool);
    VK_CHECK(vkAllocateCommandBuffers(device, &cmdInfo, &commandBuffer));

    // Render fence
    auto fenceInfo = vkw::init::FenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &renderFinishedFence));

    // Semaphores
    auto semInfo = vkw::init::SemaphoreCreateInfo();
    VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &imageReadySem));
    VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &renderFinishedSem));
}

void Frame::Cleanup() 
{
    // make sure the GPU has stopped rendering this frame
    vkWaitForFences(device, 1, &renderFinishedFence, true, 1000000000);
    // only now destroy the synchronization variables.
    vkDestroyFence(device, renderFinishedFence, nullptr);
    vkDestroySemaphore(device, imageReadySem, nullptr);
    vkDestroySemaphore(device, renderFinishedSem, nullptr);

    vkDestroyCommandPool(device, commandPool, nullptr);    
}
