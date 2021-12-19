#include "vk_wrapper/pipeline_builder.h"



VkResult vkw::GraphicsPipelineBuilder::Build(VkDevice device, VkPipeline* pPipeline) 
{
    // Viewport state
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.pNext = nullptr;

    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissors;   

    // No color blending is supported yet
    VkPipelineColorBlendStateCreateInfo blend = {};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.pNext = nullptr;

    blend.attachmentCount = 1;
    blend.pAttachments = &colorAttachment;
    blend.logicOpEnable = VK_FALSE;
    blend.logicOp = VK_LOGIC_OP_COPY;

    // Hook everything up to the pipeline info
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = nullptr;

    pipelineInfo.stageCount = (uint32_t)shaderStages.size();
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.layout = layout;
    pipelineInfo.renderPass = renderpass;

    // We don't support subpasses for now
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    VkResult res = vkCreateGraphicsPipelines(
        device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, pPipeline);
    if (res != VK_SUCCESS) {
        *pPipeline = VK_NULL_HANDLE;
    }
    return res;
}