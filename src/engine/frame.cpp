#include "engine/frame.h"
#include "engine/vk_check.h"


void Frame::Init(
        const VkDevice& device,
        uint32_t graphicsQueueFamily) 
{
    // Command pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.pNext = nullptr;

    poolInfo.queueFamilyIndex = graphicsQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &m_commandPool));

    // Command buffer
    VkCommandBufferAllocateInfo cmdInfo = { };
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdInfo.pNext = nullptr;

    cmdInfo.commandPool = m_commandPool;
    cmdInfo.commandBufferCount = 1;
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    VK_CHECK(vkAllocateCommandBuffers(device, &cmdInfo, &m_commandBuffer));
}

void Frame::Cleanup(const VkDevice& device) 
{
    vkDestroyCommandPool(device, m_commandPool, nullptr);    
}

const VkCommandBuffer& Frame::GetCommandBuffer() 
{
    return m_commandBuffer;    
}