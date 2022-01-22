#include "engine/renderer.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/shader.h"
#include "vk_wrapper/descriptor.h"
#include "vk_wrapper/pipeline_builder.h"
#include <assert.h>


void Renderer::Init(
    vkw::Device* device, 
    vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
    RenderTarget* target, VkExtent2D windowExtent, VkSurfaceKHR surface)
{
    m_device = device;
    m_descAllocator = descAllocator;
    m_descCache = descCache;
    m_target = target;
    m_windowExtent = windowExtent;
    m_surface = surface;

    InitCommands();

    // We need the swapchain to create the renderpass.
    m_swapchain.Init(m_device, m_surface, m_windowExtent);
    InitRenderPass();
    m_swapchain.CreateFramebuffers(m_renderPass);
    m_cleanupQueue.AddFunction([=] { m_swapchain.Cleanup(); });
    
    InitSynchronization();
    InitBuffers();
    InitPipeline();

    SetClearColor({ 0.0f, 0.0f, 0.0f });
}

void Renderer::Cleanup() 
{
    m_cleanupQueue.Flush();    
}

void Renderer::InitCommands() 
{
    // Queue
    vkGetDeviceQueue(m_device->logicalDevice, 
        m_device->queueFamilies.graphics, 0, &m_queue);
    
    // Command pool
    auto poolInfo = vkw::init::CommandPoolCreateInfo(
        m_device->queueFamilies.graphics);
    VK_CHECK(vkCreateCommandPool(m_device->logicalDevice, &poolInfo, nullptr, &m_cmdPool));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyCommandPool(m_device->logicalDevice, m_cmdPool, nullptr); 
    });
}

void Renderer::InitRenderPass() 
{
	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = m_swapchain.imageFormat;
	// 1 sample, we won't be doing MSAA
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	// we clear when this attachment is loaded
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	// we keep the attachment stored when the renderpass ends
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	// we don't care about stencil
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // we don't know or care about the starting layout of the attachment
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // after the renderpass ends, the image has to be on a layout ready for display
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
	// attachment number will index into the pAttachments array in the parent renderpass
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

    VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

	// connect the color attachment to the info
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	// connect the subpass to the info
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	VK_CHECK(vkCreateRenderPass(m_device->logicalDevice, &renderPassInfo, nullptr, &m_renderPass));

    m_cleanupQueue.AddFunction([=] { 
        vkDestroyRenderPass(m_device->logicalDevice, m_renderPass, nullptr);
    });
}

void Renderer::InitSynchronization()
{
    // Fence
    auto fenceInfo = vkw::init::FenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(m_device->logicalDevice, &fenceInfo, nullptr, &m_fence));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyFence(m_device->logicalDevice, m_fence, nullptr);
    });

    // Semaphores
    auto semInfo = vkw::init::SemaphoreCreateInfo();
    VK_CHECK(vkCreateSemaphore(m_device->logicalDevice, &semInfo, nullptr, &m_imageReadySem));
    VK_CHECK(vkCreateSemaphore(m_device->logicalDevice, &semInfo, nullptr, &m_renderSem));
    VK_CHECK(vkCreateSemaphore(m_device->logicalDevice, &semInfo, nullptr, &m_presentSem));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroySemaphore(m_device->logicalDevice, m_renderSem, nullptr);
        vkDestroySemaphore(m_device->logicalDevice, m_imageReadySem, nullptr);
        vkDestroySemaphore(m_device->logicalDevice, m_presentSem, nullptr);
    });
}

void Renderer::SignalRenderSem()
{
    auto submitInfo = vkw::init::SubmitInfo(nullptr, 0);
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &m_renderSem;
    VK_CHECK(vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(m_queue));
}

void Renderer::InitBuffers() 
{
    m_shaderParams.Init(m_device);
    m_shaderParams.Allocate(sizeof(ShaderParams),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_device->NameObject(m_shaderParams.buffer, "renderer params buffer");
    m_cleanupQueue.AddFunction([=] { m_shaderParams.Cleanup(); });    
}

void Renderer::InitPipeline() 
{    
    // Compile the shaders
    vkw::ShaderCompiler compiler(m_device, "/home/mathis/src/f-rep-voxel-rendering/shaders/");
    VkShaderModule vertShader = compiler.Compile(
        "renderer/triangle.vert", vkw::ShaderCompiler::Stage::VERT);
    VkShaderModule fragShader = compiler.Compile(
        "renderer/triangle.frag", vkw::ShaderCompiler::Stage::FRAG);
    
    // Descriptor Sets
    m_descSets = std::vector<VkDescriptorSet>(1);
    auto dSetLayouts = std::vector<VkDescriptorSetLayout>(1);

    // Descriptor Set 0    
    auto imageInfo = vkw::init::DescriptorImageInfo(
        //m_sampler, m_imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_target->sampler, m_target->view, VK_IMAGE_LAYOUT_GENERAL);
    auto paramsInfo = vkw::init::DescriptorBufferInfo(
        m_shaderParams.buffer, 0, m_shaderParams.size);
    vkw::DescriptorBuilder(m_descCache, m_descAllocator)
        .BindImage(0, &imageInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
        .BindBuffer(1, &paramsInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
        .Build(&m_descSets[0], &dSetLayouts[0]);
    m_device->NameObject(m_descSets[0], "renderer descriptor set 0");

    // Pipeline layout
    auto layoutInfo = vkw::init::PipelineLayoutCreateInfo();
    layoutInfo.setLayoutCount = (uint32_t)dSetLayouts.size();
    layoutInfo.pSetLayouts = dSetLayouts.data();
    VK_CHECK(vkCreatePipelineLayout(m_device->logicalDevice, &layoutInfo, nullptr, &m_pipelineLayout));

    // Create the pipeline
    vkw::GraphicsPipelineBuilder builder;
    builder.shaderStages.push_back(
        vkw::init::PipelineShaderStageCreateInfo(
            VK_SHADER_STAGE_VERTEX_BIT, vertShader));
    builder.shaderStages.push_back(
        vkw::init::PipelineShaderStageCreateInfo(
            VK_SHADER_STAGE_FRAGMENT_BIT, fragShader));
    
    builder.vertexInputInfo = vkw::init::PipelineVertexInputStateCreateInfo();
    builder.inputAssembly = vkw::init::PipelineInputAssemblyStateCreateInfo(
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    
    builder.viewport.x = 0.0f;
    builder.viewport.y = 0.0f;
    builder.viewport.width = (float)m_windowExtent.width;
    builder.viewport.height = (float)m_windowExtent.height;
    builder.viewport.minDepth = 0.0f;
    builder.viewport.maxDepth = 1.0f;

    builder.scissors.offset = { 0, 0 };
    builder.scissors.extent = m_windowExtent;

    builder.rasterizer = vkw::init::PipelineRasterizationStateCreateInfo(
        VK_POLYGON_MODE_FILL);
    builder.multisampling = vkw::init::PipelineMultisampleStateCreateInfo();
    builder.colorAttachment = vkw::init::PipelineColorBlendAttachmentState(
        VK_COLOR_COMPONENT_R_BIT | 
        VK_COLOR_COMPONENT_G_BIT | 
        VK_COLOR_COMPONENT_B_BIT | 
        VK_COLOR_COMPONENT_A_BIT);

    builder.layout = m_pipelineLayout;
    builder.renderpass = m_renderPass;

    VkResult res = builder.Build(m_device->logicalDevice, &m_pipeline);
    if (res != VK_SUCCESS) {
        printf("[-] Error building graphics pipeline\n");
        assert(false);
    }
    m_device->NameObject(m_pipeline, "renderer pipeline");

    // We can destroy the shaders right after creating the pipeline.
    vkDestroyShaderModule(m_device->logicalDevice, vertShader, nullptr);
    vkDestroyShaderModule(m_device->logicalDevice, fragShader, nullptr);
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyPipelineLayout(m_device->logicalDevice, m_pipelineLayout, nullptr);
        vkDestroyPipeline(m_device->logicalDevice, m_pipeline, nullptr); 
    });
}

void Renderer::BeginFrame() 
{
    // Acquire a swapchain image.
    m_swapCurrImg = m_swapchain.AcquireNewImage(m_imageReadySem);
}


void Renderer::UpdateShaderParams() 
{
    ShaderParams* params = (ShaderParams*)m_shaderParams.Map();   
    params->temporalSampleCount = m_target->temporalSampleCount;
    m_shaderParams.Unmap();
}

void Renderer::RecordRenderCmd(VkCommandBuffer cmd) 
{
    /*m_image.ChangeLayout(cmd,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, VK_ACCESS_SHADER_READ_BIT);*/

    // Begin renderpass
    auto rpInfo = vkw::init::RenderPassBeginInfo(
        m_renderPass, 
        m_swapchain.framebuffers[m_swapCurrImg], 
        m_swapchain.windowExtent,
        &m_clearValue);
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipelineLayout, 
        0, m_descSets.size(), m_descSets.data(), 
        0, nullptr);
    // Draw the 6 vertices of the quad.
    vkCmdDraw(cmd, 6, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
}

void Renderer::SubmitRenderCmd(VkCommandBuffer cmd, VkSemaphore waitSem) 
{
    auto info = vkw::init::SubmitInfo(&cmd);

    VkSemaphore waitSems[] = { waitSem, m_imageReadySem };
    VkPipelineStageFlags waitDstMasks[] = { 
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    info.waitSemaphoreCount = 2;
    info.pWaitSemaphores = &waitSems[0];
    info.pWaitDstStageMask = &waitDstMasks[0];

    VkSemaphore signalSemaphores[] = { m_renderSem, m_presentSem };
    info.signalSemaphoreCount = 2;
    info.pSignalSemaphores = &signalSemaphores[0];
    
    VK_CHECK(vkQueueSubmit(m_queue, 1, &info, m_fence));
}

void Renderer::Render(VkSemaphore waitSem) 
{
    // Wait for the previous command to finish.
    VK_CHECK(vkWaitForFences(m_device->logicalDevice, 1, &m_fence, VK_TRUE, 1000000000));
    VK_CHECK(vkResetFences(m_device->logicalDevice, 1, &m_fence));

    UpdateShaderParams();

    m_device->QueueBeginLabel(m_queue, "render");
    {
        // Reset the command pool (and its buffers).
        VK_CHECK(vkResetCommandPool(m_device->logicalDevice, m_cmdPool, 0));
        // Allocate the command buffer.
        VkCommandBuffer cmd;
        auto allocInfo = vkw::init::CommandBufferAllocateInfo(m_cmdPool);
        VK_CHECK(vkAllocateCommandBuffers(
            m_device->logicalDevice, &allocInfo, &cmd));
        // Begin the command.
        auto beginInfo = vkw::init::CommandBufferBeginInfo(
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
        // Record the command
        RecordRenderCmd(cmd);
        // End the command
        VK_CHECK(vkEndCommandBuffer(cmd));
        // Submit
        SubmitRenderCmd(cmd, waitSem);
    }
    m_device->QueueEndLabel(m_queue);
}

void Renderer::EndFrame() 
{
    auto presentInfo = vkw::init::PresentInfoKHR(
        &m_swapchain.swapchain, &m_swapCurrImg);
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &m_presentSem;
    VK_CHECK(vkQueuePresentKHR(m_queue, &presentInfo));  
}

void Renderer::SetClearColor(glm::vec3 color) 
{
    m_clearValue.color.float32[0] = color.x;
    m_clearValue.color.float32[1] = color.y;
    m_clearValue.color.float32[2] = color.z;
    m_clearValue.color.float32[3] = 1.0f;
}