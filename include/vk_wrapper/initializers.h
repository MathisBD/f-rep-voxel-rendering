#pragma once
#include <vulkan/vulkan.h>
#include <string>


namespace vkw
{
namespace init
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

VkDescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(
    uint32_t bindingCount,
    const VkDescriptorSetLayoutBinding* pBindings,
    VkDescriptorSetLayoutCreateFlags flags = 0);

VkDescriptorPoolCreateInfo DescriptorPoolCreateInfo(
    uint32_t maxSets,
    uint32_t poolSizeCount,
    VkDescriptorPoolSize* pPoolSizes,
    VkDescriptorPoolCreateFlags flags = 0);

VkShaderModuleCreateInfo ShaderModuleCreateInfo(
    uint32_t codeSizeBytes,
    uint32_t* pCode,
    VkShaderModuleCreateFlags flags = 0);

VkPipelineShaderStageCreateInfo PipelineShaderStageCreateInfo(
    VkShaderStageFlagBits stage, 
    VkShaderModule shaderModule,
    VkPipelineShaderStageCreateFlags flags = 0);

VkPipelineVertexInputStateCreateInfo PipelineVertexInputStateCreateInfo(
    VkPipelineVertexInputStateCreateFlags flags = 0); 

VkPipelineInputAssemblyStateCreateInfo PipelineInputAssemblyStateCreateInfo(
    VkPrimitiveTopology topology,
    VkPipelineInputAssemblyStateCreateFlags flags = 0);

VkPipelineRasterizationStateCreateInfo PipelineRasterizationStateCreateInfo(
    VkPolygonMode polygonMode,
    VkPipelineRasterizationStateCreateFlags flags = 0);

VkPipelineMultisampleStateCreateInfo PipelineMultisampleStateCreateInfo();

VkPipelineColorBlendAttachmentState PipelineColorBlendAttachmentState(
    VkColorComponentFlags colorWriteMask);

VkPipelineLayoutCreateInfo PipelineLayoutCreateInfo();

VkDescriptorBufferInfo DescriptorBufferInfo(
    VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);

VkCommandBufferBeginInfo CommandBufferBeginInfo(
    VkCommandBufferUsageFlags flags = 0);

VkSubmitInfo SubmitInfo(
    VkCommandBuffer* pCmds, uint32_t cmdCount = 1);

VkImageViewCreateInfo ImageViewCreateInfo(
    VkFormat format, VkImage image, VkImageAspectFlags aspectFlags);

VkDescriptorImageInfo DescriptorImageInfo(
    VkSampler sampler, VkImageView view, VkImageLayout layout);

VkSamplerCreateInfo SamplerCreateInfo(
    VkFilter filters, VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT);

VkRenderPassBeginInfo RenderPassBeginInfo(
    VkRenderPass renderpass, VkFramebuffer framebuffer, VkExtent2D extent, const VkClearValue* pClear);

VkPresentInfoKHR PresentInfoKHR(
    const VkSwapchainKHR* swapchain, const uint32_t* pSwapchainImgIdx);

}   // init

}   // vkw