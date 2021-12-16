#pragma once
#include <vulkan/vulkan.h>
#include <vector>



namespace vkw
{
    class GraphicsPipelineBuilder
    {
    public:
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
        VkPipelineVertexInputStateCreateInfo vertexInputInfo;
        VkPipelineInputAssemblyStateCreateInfo inputAssembly;
        VkViewport viewport;
        VkRect2D scissors;
        VkPipelineRasterizationStateCreateInfo rasterizer;
        VkPipelineColorBlendAttachmentState colorAttachment;
        VkPipelineMultisampleStateCreateInfo multisampling;
        VkPipelineLayout layout;
        VkRenderPass renderpass;

        VkResult Build(VkDevice device, VkPipeline* pPipeline);
    };
}