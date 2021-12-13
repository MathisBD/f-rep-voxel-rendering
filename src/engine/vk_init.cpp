#include "engine/vk_init.h"
#include <vulkan/vulkan.h>


VkCommandPoolCreateInfo vkinit::CommandPoolCreateInfo(
    uint32_t queueFamilyIndex, 
    VkCommandPoolCreateFlags flags /*= 0*/) 
{
    VkCommandPoolCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.pNext = nullptr;

    info.queueFamilyIndex = queueFamilyIndex;
    info.flags = flags;
    
    return info;
}

VkCommandBufferAllocateInfo vkinit::CommandBufferAllocateInfo(
    VkCommandPool pool, 
    uint32_t count /*= 1*/, 
    VkCommandBufferLevel level /*= VK_COMMAND_BUFFER_LEVEL_PRIMARY*/) 
{
    VkCommandBufferAllocateInfo info = { };
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.pNext = nullptr;

    info.commandPool = pool;
    info.commandBufferCount = count;
    info.level = level;
    
    return info;
}

VkFenceCreateInfo vkinit::FenceCreateInfo(VkFenceCreateFlags flags /*= 0*/) 
{
    VkFenceCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	info.pNext = nullptr;

	info.flags = flags;

    return info;
}

VkSemaphoreCreateInfo vkinit::SemaphoreCreateInfo(VkSemaphoreCreateFlags flags /*= 0*/) 
{
    VkSemaphoreCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	info.pNext = nullptr;
	
    info.flags = flags;
    
    return info;
}