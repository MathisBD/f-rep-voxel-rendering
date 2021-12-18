#include "application.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/shader.h"
#include "vk_wrapper/pipeline_builder.h"


void Application::Init()
{
    EngineBase::Init();

    InitBuffer();
    InitImage();
    InitPipelines();
}

void Application::InitBuffer() 
{
    // allocate the camera buffer
    m_cameraBuffer.Init(m_vmaAllocator);
    m_cameraBuffer.Allocate(
        sizeof(GPUCameraData), 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] { m_cameraBuffer.Cleanup(); });    
}

void Application::InitImage() 
{
    // Allocate the image
    m_image.Init(m_vmaAllocator);
    m_image.Allocate(
        { .width = 256, .height = 256 },
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    m_cleanupQueue.AddFunction([=] { m_image.Cleanup(); });
    
    // Create the image contents in a CPU staging buffer.
    vkw::Buffer stagingBuffer;
    stagingBuffer.Init(m_vmaAllocator);
    stagingBuffer.Allocate(
        m_image.extent.width * m_image.extent.height * sizeof(float) * 4,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_ONLY);
    float* buf = (float*)stagingBuffer.Map();
    for (size_t y = 0; y < m_image.extent.height; y++) {
        for (size_t x = 0; x < m_image.extent.width; x++) {
            size_t idx = 4 * (x + y * m_image.extent.width);
            if (x + y < 256) {
                buf[idx] = buf[idx+1] = buf[idx+2] = 0.0f;
            }
            else {
                buf[idx] = buf[idx+1] = buf[idx+2] = 1.0f;
            }
            buf[idx+3] = 1.0f;
}
    }
    stagingBuffer.Unmap();

    // Copy the image contents.
    ImmediateSubmit([=] (VkCommandBuffer cmd) {
        m_image.ChangeLayout(cmd, 
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, VK_ACCESS_TRANSFER_WRITE_BIT);
        m_image.CopyFromBuffer(cmd, &stagingBuffer);
        m_image.ChangeLayout(cmd,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    });

    // Free the staging buffer
    stagingBuffer.Cleanup();
}

void Application::InitPipelines() 
{    
    // Load the shaders
    vkw::Shader vertexShader, fragmentShader;
    vertexShader.Init(m_device.logicalDevice, "../shaders/triangle.vert.spv");
    fragmentShader.Init(m_device.logicalDevice, "../shaders/triangle.frag.spv");

    // Descriptor Sets
    m_dSets = std::vector<VkDescriptorSet>(2);
    auto dSetLayouts = std::vector<VkDescriptorSetLayout>(2);

    // Descriptor Set 0
    auto cameraBufferInfo = vkw::init::DescriptorBufferInfo(
        m_cameraBuffer.buffer, 0, sizeof(GPUCameraData));
    vkw::DescriptorBuilder(&m_descriptorCache, &m_descriptorAllocator)
        .BindBuffer(0, &cameraBufferInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
        .Build(&m_dSets[0], &dSetLayouts[0]);

    // Descriptor Set 1
    auto samplerInfo = vkw::init::SamplerCreateInfo(VK_FILTER_NEAREST);
    VkSampler sampler;
    VK_CHECK(vkCreateSampler(
        m_device.logicalDevice, &samplerInfo, nullptr, &sampler));
    m_cleanupQueue.AddFunction([=] {
        vkDestroySampler(m_device.logicalDevice, sampler, nullptr);
    });

    auto viewInfo = vkw::init::ImageViewCreateInfo(
        m_image.format, m_image.image, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageView imageView;
    VK_CHECK(vkCreateImageView(m_device.logicalDevice, &viewInfo, nullptr, &imageView));
    m_cleanupQueue.AddFunction([=] {
        vkDestroyImageView(m_device.logicalDevice, imageView, nullptr);
    });

    auto imageInfo = vkw::init::DescriptorImageInfo(
        sampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkw::DescriptorBuilder(&m_descriptorCache, &m_descriptorAllocator)
        .BindImage(0, &imageInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
        .Build(&m_dSets[1], &dSetLayouts[1]);
    // Pipeline layout
    auto layoutInfo = vkw::init::PipelineLayoutCreateInfo();
    layoutInfo.setLayoutCount = (uint32_t)dSetLayouts.size();
    layoutInfo.pSetLayouts = dSetLayouts.data();
    VK_CHECK(vkCreatePipelineLayout(m_device.logicalDevice, &layoutInfo, nullptr, &m_pipelineLayout));

    // Create the pipeline
    vkw::GraphicsPipelineBuilder builder;
    builder.shaderStages.push_back(
        vkw::init::PipelineShaderStageCreateInfo(
            VK_SHADER_STAGE_VERTEX_BIT, vertexShader.shader));
    builder.shaderStages.push_back(
        vkw::init::PipelineShaderStageCreateInfo(
            VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader.shader));
    
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
    builder.renderpass = m_renderer.swapchain.renderPass;

    VkResult res = builder.Build(m_device.logicalDevice, &m_pipeline);
    if (res != VK_SUCCESS) {
        printf("[-] Error building graphics pipeline\n");
        assert(false);
    }

    // We can destroy the shaders right after creating the pipeline.
    vertexShader.Cleanup();
    fragmentShader.Cleanup();
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyPipelineLayout(m_device.logicalDevice, m_pipelineLayout, nullptr);
        vkDestroyPipeline(m_device.logicalDevice, m_pipeline, nullptr); 
    });
}

void Application::Draw() 
{
    // Upload the camera data
    GPUCameraData* data = (GPUCameraData*)m_cameraBuffer.Map();
    data->color = { 0.0f, 1.0f, 1.0f, 0.0f };
    m_cameraBuffer.Unmap();

    // Draw the frame
    Renderer::DrawInfo drawInfo = {};
    drawInfo.pipeline = m_pipeline;
    drawInfo.pipelineLayout = m_pipelineLayout;
    drawInfo.descriptorSets = m_dSets;
    m_renderer.Draw(&drawInfo);    
}