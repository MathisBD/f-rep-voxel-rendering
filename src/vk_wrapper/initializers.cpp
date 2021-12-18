#include "vk_wrapper/initializers.h"


VkDeviceQueueCreateInfo vkw::init::DeviceQueueCreateInfo(
    uint32_t queueFamily,
    float queuePriority /*= 0.0f*/) 
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

VkDescriptorPoolCreateInfo vkw::init::DescriptorPoolCreateInfo(
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

VkPipelineShaderStageCreateInfo vkw::init::PipelineShaderStageCreateInfo(
    VkShaderStageFlagBits stage, 
    VkShaderModule shaderModule,
    VkPipelineShaderStageCreateFlags flags /*= 0*/) 
{
    VkPipelineShaderStageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.pNext = nullptr;

    info.stage = stage;
    info.module = shaderModule;
    info.flags = flags;
    info.pName = "main";

    return info;
}

VkPipelineVertexInputStateCreateInfo vkw::init::PipelineVertexInputStateCreateInfo(
    VkPipelineVertexInputStateCreateFlags flags /*= 0*/) 
{
    VkPipelineVertexInputStateCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    info.pNext = nullptr;    

    info.vertexAttributeDescriptionCount = 0;
    info.vertexBindingDescriptionCount = 0;
    info.flags = flags;

    return info;
}

VkPipelineInputAssemblyStateCreateInfo vkw::init::PipelineInputAssemblyStateCreateInfo(
    VkPrimitiveTopology topology,
    VkPipelineInputAssemblyStateCreateFlags flags /*= 0*/) 
{
    VkPipelineInputAssemblyStateCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    info.pNext = nullptr;

    info.topology = topology;
    info.flags = flags;    
    info.primitiveRestartEnable = VK_FALSE;
    
    return info;
}

VkPipelineRasterizationStateCreateInfo vkw::init::PipelineRasterizationStateCreateInfo(
    VkPolygonMode polygonMode,
    VkPipelineRasterizationStateCreateFlags flags /*= 0*/) 
{
    VkPipelineRasterizationStateCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	info.pNext = nullptr;

    // No depth bias/clamp.
    info.depthBiasEnable = VK_FALSE;
    info.depthClampEnable = VK_FALSE;
    info.rasterizerDiscardEnable = VK_FALSE;

    // No culling.
    info.cullMode = VK_CULL_MODE_NONE;
    info.frontFace = VK_FRONT_FACE_CLOCKWISE;

    info.polygonMode = polygonMode;
	info.lineWidth = 1.0f;
    info.flags = flags;

    return info;
}

VkPipelineMultisampleStateCreateInfo vkw::init::PipelineMultisampleStateCreateInfo() 
{
    VkPipelineMultisampleStateCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    info.pNext = nullptr;
    
    // Default : no multisampling.
    info.sampleShadingEnable = VK_FALSE;
    info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    info.minSampleShading = 1.0f;
    info.pSampleMask = nullptr;
    info.alphaToOneEnable = VK_FALSE;
    info.alphaToCoverageEnable = VK_FALSE;
    info.flags = 0;

    return info;
}

VkPipelineColorBlendAttachmentState vkw::init::PipelineColorBlendAttachmentState(
    VkColorComponentFlags colorWriteMask) 
{
    VkPipelineColorBlendAttachmentState attachment = {};
    
    attachment.colorWriteMask = colorWriteMask;
    attachment.blendEnable = VK_FALSE;
    
    return attachment;
}

VkPipelineLayoutCreateInfo vkw::init::PipelineLayoutCreateInfo() 
{
    VkPipelineLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.pNext = nullptr;
    return info;
}

VkDescriptorBufferInfo vkw::init::DescriptorBufferInfo(
    VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) 
{
    VkDescriptorBufferInfo info = {};
    info.buffer = buffer;
    info.offset = offset;
    info.range = range;
    return info;
}

VkCommandBufferBeginInfo vkw::init::CommandBufferBeginInfo(
    VkCommandBufferUsageFlags flags /*= 0*/) 
{
    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.pNext = nullptr;

    info.pInheritanceInfo = nullptr;
    info.flags = flags;
    return info;
}

VkSubmitInfo vkw::init::SubmitInfo(
    VkCommandBuffer* pCmds,
    uint32_t cmdCount /*= 1*/) 
{
    VkSubmitInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	info.pNext = nullptr;

	info.commandBufferCount = cmdCount;
	info.pCommandBuffers = pCmds;

    return info;
}

VkImageViewCreateInfo vkw::init::ImageViewCreateInfo(
    VkFormat format, VkImage image, VkImageAspectFlags aspectFlags) 
{
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.pNext = nullptr;

    info.format = format;
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;

    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount = 1;
    info.subresourceRange.baseMipLevel = 0;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.aspectMask = aspectFlags;

    return info;
}

VkDescriptorImageInfo vkw::init::DescriptorImageInfo(
    VkSampler sampler, VkImageView view, VkImageLayout layout) 
{
    VkDescriptorImageInfo info = {};
    info.sampler = sampler;
    info.imageView = view;
    info.imageLayout = layout;
    return info;
}

VkSamplerCreateInfo vkw::init::SamplerCreateInfo(
    VkFilter filters, VkSamplerAddressMode addressMode /*= VK_SAMPLER_ADDRESS_MODE_REPEAT*/) 
{
    VkSamplerCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.pNext = nullptr;

    info.magFilter = filters;
    info.minFilter = filters;
    info.addressModeU = addressMode;
    info.addressModeV = addressMode;
    info.addressModeW = addressMode;
    return info;
}