#pragma once
#include <vulkan/vulkan.h>


namespace vkinit
{
    VkCommandPoolCreateInfo CommandPoolCreateInfo(
        uint32_t queueFamilyIndex, 
        VkCommandPoolCreateFlags flags = 0);
    
    VkCommandBufferAllocateInfo CommandBufferAllocateInfo(
        VkCommandPool pool, 
        uint32_t count = 1, 
        VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    VkFenceCreateInfo FenceCreateInfo(VkFenceCreateFlags flags = 0);
    
    VkSemaphoreCreateInfo SemaphoreCreateInfo(VkSemaphoreCreateFlags flags = 0);
}