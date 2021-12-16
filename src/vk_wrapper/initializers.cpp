#include "vk_wrapper/initializers.h"


VkDeviceQueueCreateInfo vkw::init::DeviceQueueCreateInfo(
    uint32_t queueFamily,
    float queuePriority = 0.0f) 
{
    VkDeviceQueueCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    info.pNext = nullptr;

    info.queueFamilyIndex = queueFamily;
    info.queueCount = 1;
    info.pQueuePriorities = &queuePriority;

    return info;    
}


VkCommandPoolCreateInfo vkw::init::CommandPoolCreateInfo(
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

VkCommandBufferAllocateInfo vkw::init::CommandBufferAllocateInfo(
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

VkFenceCreateInfo vkw::init::FenceCreateInfo(VkFenceCreateFlags flags /*= 0*/) 
{
    VkFenceCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	info.pNext = nullptr;

	info.flags = flags;
    return info;
}

VkSemaphoreCreateInfo vkw::init::SemaphoreCreateInfo(VkSemaphoreCreateFlags flags /*= 0*/) 
{
    VkSemaphoreCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	info.pNext = nullptr;
	
    info.flags = flags;
    return info;
}

VkDescriptorSetLayoutCreateInfo vkw::init::DescriptorSetLayoutCreateInfo(
    uint32_t bindingCount,
    const VkDescriptorSetLayoutBinding* pBindings,
    VkDescriptorSetLayoutCreateFlags flags /*= 0*/) 
{
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.pNext = nullptr;

    info.bindingCount = bindingCount;
    info.pBindings = pBindings;
    info.flags = flags;
    return info;
}

VkDescriptorPoolCreateInfo DescriptorPoolCreateInfo(
    uint32_t maxSets,
    uint32_t poolSizeCount,
    VkDescriptorPoolSize* pPoolSizes,
    VkDescriptorPoolCreateFlags flags /*= 0*/) 
{
    VkDescriptorPoolCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.pNext = nullptr;

    info.maxSets = maxSets;
    info.poolSizeCount = poolSizeCount;
    info.pPoolSizes = pPoolSizes;
    info.flags = flags;
    return info;
}

VkShaderModuleCreateInfo vkw::init::ShaderModuleCreateInfo(
    uint32_t codeSizeBytes,
    uint32_t* pCode,
    VkShaderModuleCreateFlags flags /*= 0*/) 
{
    VkShaderModuleCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.pNext = nullptr;
        
    info.codeSize = codeSizeBytes;
    info.pCode = pCode;
    info.flags = flags;

    return info;
}