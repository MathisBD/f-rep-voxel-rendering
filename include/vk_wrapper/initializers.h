#pragma once
#include <vulkan/vulkan.h>


namespace vkw
{
namespace init
{

VkDeviceQueueCreateInfo DeviceQueueCreateInfo(
    uint32_t queueFamily,
    float queuePriority = 0.0f);

VkCommandPoolCreateInfo CommandPoolCreateInfo(
    uint32_t queueFamilyIndex, 
    VkCommandPoolCreateFlags flags = 0);

VkCommandBufferAllocateInfo CommandBufferAllocateInfo(
    VkCommandPool pool, 
    uint32_t count = 1, 
    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

VkFenceCreateInfo FenceCreateInfo(VkFenceCreateFlags flags = 0);

VkSemaphoreCreateInfo SemaphoreCreateInfo(VkSemaphoreCreateFlags flags = 0);

VkDescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(
    uint32_t bindingCount,
    const VkDescriptorSetLayoutBinding* pBindings,
    VkDescriptorSetLayoutCreateFlags flags = 0);

VkDescriptorPoolCreateInfo DescriptorPoolCreateInfo(
    uint32_t maxSets,
    uint32_t poolSizeCount,
    VkDescriptorPoolSize* pPoolSizes,
    VkDescriptorPoolCreateFlags flags = 0);

}   // init

}   // vkw