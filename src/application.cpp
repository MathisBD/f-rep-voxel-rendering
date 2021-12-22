#include "application.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/shader.h"
#include "vk_wrapper/pipeline_builder.h"


void Application::Init()
{
    EngineBase::Init();

    printf("[+] Queue families :\n\tgraphics=%u\n\tcompute=%u\n\ttransfer=%u\n",
        m_device.queueFamilies.graphics, 
        m_device.queueFamilies.compute,
        m_device.queueFamilies.transfer);

    InitImage();
    InitGraphics();
    InitGraphicsPipeline();
    InitCompute();
    InitComputePipeline();
}

void Application::InitImage() 
{
    // Allocate the image
    m_image.Init(m_vmaAllocator);
    std::vector<uint32_t> queueFamilies;
    VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (m_device.queueFamilies.graphics != m_device.queueFamilies.compute) {
        queueFamilies = { 
            m_device.queueFamilies.graphics,
            m_device.queueFamilies.compute };
        sharingMode = VK_SHARING_MODE_CONCURRENT;
    }
    m_image.Allocate(
        m_windowExtent,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        sharingMode,
        &queueFamilies);
    m_cleanupQueue.AddFunction([=] { m_image.Cleanup(); });

    // Set the image layout (GENERAL).
    ImmediateSubmit(
        [=] (VkCommandBuffer cmd) { 
            m_image.ChangeLayout(cmd, 
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, VK_ACCESS_SHADER_WRITE_BIT);
        });

    // Image view
    auto viewInfo = vkw::init::ImageViewCreateInfo(
        m_image.format, m_image.image, VK_IMAGE_ASPECT_COLOR_BIT);
    VK_CHECK(vkCreateImageView(m_device.logicalDevice, &viewInfo, nullptr, &m_imageView));
    m_cleanupQueue.AddFunction([=] {
        vkDestroyImageView(m_device.logicalDevice, m_imageView, nullptr);
    });

    // Sampler
    auto samplerInfo = vkw::init::SamplerCreateInfo(VK_FILTER_NEAREST);
    VK_CHECK(vkCreateSampler(
        m_device.logicalDevice, &samplerInfo, nullptr, &m_sampler));
    m_cleanupQueue.AddFunction([=] {
        vkDestroySampler(m_device.logicalDevice, m_sampler, nullptr);
    });
}

void Application::InitGraphics() 
{
    // Swapchain 
    m_graphics.swapchain.Init(&m_device, m_surface, m_windowExtent);
    m_cleanupQueue.AddFunction([=] { m_graphics.swapchain.Cleanup(); });

    // Queue
    vkGetDeviceQueue(m_device.logicalDevice, 
        m_device.queueFamilies.graphics, 0, &m_graphics.queue);
    
    // Command pool
    auto poolInfo = vkw::init::CommandPoolCreateInfo(
        m_device.queueFamilies.graphics);
    VK_CHECK(vkCreateCommandPool(m_device.logicalDevice, &poolInfo, nullptr, &m_graphics.cmdPool));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyCommandPool(m_device.logicalDevice, m_graphics.cmdPool, nullptr); 
    });

    // Synchronization
    auto fenceInfo = vkw::init::FenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(m_device.logicalDevice, &fenceInfo, nullptr, &m_graphics.fence));
    
    auto semInfo = vkw::init::SemaphoreCreateInfo();
    VK_CHECK(vkCreateSemaphore(m_device.logicalDevice, &semInfo, nullptr, &m_graphics.imageReadySem));
    VK_CHECK(vkCreateSemaphore(m_device.logicalDevice, &semInfo, nullptr, &m_graphics.semaphore));
    VK_CHECK(vkCreateSemaphore(m_device.logicalDevice, &semInfo, nullptr, &m_graphics.presentSem));
    
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyFence(m_device.logicalDevice, m_graphics.fence, nullptr);
        vkDestroySemaphore(m_device.logicalDevice, m_graphics.semaphore, nullptr);
        vkDestroySemaphore(m_device.logicalDevice, m_graphics.imageReadySem, nullptr);
        vkDestroySemaphore(m_device.logicalDevice, m_graphics.presentSem, nullptr);
    });

    // Signal the graphics to compute semaphore
    auto submitInfo = vkw::init::SubmitInfo(nullptr, 0);
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &m_graphics.semaphore;
    VK_CHECK(vkQueueSubmit(m_graphics.queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(m_graphics.queue));
}

void Application::InitGraphicsPipeline() 
{    
    // Load the shaders
    vkw::Shader vertexShader, fragmentShader;
    vertexShader.Init(m_device.logicalDevice, "../shaders/triangle.vert.spv");
    fragmentShader.Init(m_device.logicalDevice, "../shaders/triangle.frag.spv");

    // Descriptor Sets
    m_graphics.dSets = std::vector<VkDescriptorSet>(1);
    auto dSetLayouts = std::vector<VkDescriptorSetLayout>(1);

    // Descriptor Set 0    
    auto imageInfo = vkw::init::DescriptorImageInfo(
        //m_sampler, m_imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_sampler, m_imageView, VK_IMAGE_LAYOUT_GENERAL);
    vkw::DescriptorBuilder(&m_descriptorCache, &m_descriptorAllocator)
        .BindImage(0, &imageInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
        .Build(&m_graphics.dSets[0], &dSetLayouts[0]);
    
    // Pipeline layout
    auto layoutInfo = vkw::init::PipelineLayoutCreateInfo();
    layoutInfo.setLayoutCount = (uint32_t)dSetLayouts.size();
    layoutInfo.pSetLayouts = dSetLayouts.data();
    VK_CHECK(vkCreatePipelineLayout(m_device.logicalDevice, &layoutInfo, nullptr, &m_graphics.pipelineLayout));

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

    builder.layout = m_graphics.pipelineLayout;
    builder.renderpass = m_graphics.swapchain.renderPass;

    VkResult res = builder.Build(m_device.logicalDevice, &m_graphics.pipeline);
    if (res != VK_SUCCESS) {
        printf("[-] Error building graphics pipeline\n");
        assert(false);
    }

    // We can destroy the shaders right after creating the pipeline.
    vertexShader.Cleanup();
    fragmentShader.Cleanup();
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyPipelineLayout(m_device.logicalDevice, m_graphics.pipelineLayout, nullptr);
        vkDestroyPipeline(m_device.logicalDevice, m_graphics.pipeline, nullptr); 
    });
}

void Application::InitCompute() 
{
    // Queue
    vkGetDeviceQueue(m_device.logicalDevice, 
        m_device.queueFamilies.compute, 0, &m_compute.queue);
    
    // Command pool
    auto poolInfo = vkw::init::CommandPoolCreateInfo(
        m_device.queueFamilies.compute);
    VK_CHECK(vkCreateCommandPool(m_device.logicalDevice, &poolInfo, nullptr, &m_compute.cmdPool));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyCommandPool(m_device.logicalDevice, m_compute.cmdPool, nullptr); 
    });

    // Synchronization
    auto fenceInfo = vkw::init::FenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(m_device.logicalDevice, &fenceInfo, nullptr, &m_compute.fence));
    
    auto semInfo = vkw::init::SemaphoreCreateInfo();
    VK_CHECK(vkCreateSemaphore(m_device.logicalDevice, &semInfo, nullptr, &m_compute.semaphore));
    
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyFence(m_device.logicalDevice, m_compute.fence, nullptr);
        vkDestroySemaphore(m_device.logicalDevice, m_compute.semaphore, nullptr);
    });

    // Uniform buffer
    m_compute.ddaUniforms.Init(m_vmaAllocator);
    m_compute.ddaUniforms.Allocate(
        sizeof(DDAUniforms), 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] { m_compute.ddaUniforms.Cleanup(); });

    // Lights buffer
    m_compute.ddaLights.Init(m_vmaAllocator);
    m_compute.ddaLights.Allocate(
        sizeof(glm::vec4) + m_compute.lightCount * sizeof(DDALight),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] { m_compute.ddaLights.Cleanup(); });

    // Voxels buffer
    size_t res = m_compute.gridResolution;
    m_compute.ddaVoxels.Init(m_vmaAllocator);
    m_compute.ddaVoxels.Allocate(
        res * res * res * sizeof(DDAVoxel),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] { m_compute.ddaVoxels.Cleanup(); });

    // We only upload the voxels once.
    UpdateDDAVoxels();

    // Camera
    m_compute.camera.position = { 0, 0, 40.0f };
    m_compute.camera.forward = { 0, 0, -1 };
    m_compute.camera.initialUp = { 0, 1, 0 };
    m_compute.camera.fovDeg = 45;
    m_compute.camera.aspectRatio = m_windowExtent.width / (float)m_windowExtent.height;
    m_compute.camera.Init();
}

void Application::InitComputePipeline() 
{
    // Load the shader
    vkw::Shader shader;
    shader.Init(m_device.logicalDevice, "../shaders/dda.comp.spv");
    
    // Descriptor Sets
    m_compute.dSets = std::vector<VkDescriptorSet>(1);
    auto dSetLayouts = std::vector<VkDescriptorSetLayout>(1);

    // Descriptor Set 0    
    auto outImageInfo = vkw::init::DescriptorImageInfo(
        m_sampler, m_imageView, VK_IMAGE_LAYOUT_GENERAL);
    auto ddaUniformsInfo = vkw::init::DescriptorBufferInfo(
        m_compute.ddaUniforms.buffer, 0, m_compute.ddaUniforms.size);
    auto ddaVoxelsInfo = vkw::init::DescriptorBufferInfo(
        m_compute.ddaVoxels.buffer, 0, m_compute.ddaVoxels.size);
    auto ddaLightsInfo = vkw::init::DescriptorBufferInfo(
        m_compute.ddaLights.buffer, 0, m_compute.ddaLights.size);
    vkw::DescriptorBuilder(&m_descriptorCache, &m_descriptorAllocator)
        .BindImage(0, &outImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(1, &ddaUniformsInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(2, &ddaVoxelsInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(3, &ddaLightsInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .Build(&m_compute.dSets[0], &dSetLayouts[0]);

    // Pipeline layout
    auto layoutInfo = vkw::init::PipelineLayoutCreateInfo();
    layoutInfo.setLayoutCount = (uint32_t)dSetLayouts.size();
    layoutInfo.pSetLayouts = dSetLayouts.data();
    VK_CHECK(vkCreatePipelineLayout(m_device.logicalDevice, &layoutInfo, nullptr, &m_compute.pipelineLayout));
 
    // Pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = nullptr;
    pipelineInfo.layout = m_compute.pipelineLayout;
    pipelineInfo.stage = vkw::init::PipelineShaderStageCreateInfo(
        VK_SHADER_STAGE_COMPUTE_BIT, shader.shader);
    VK_CHECK(vkCreateComputePipelines(m_device.logicalDevice, VK_NULL_HANDLE, 
        1, &pipelineInfo, nullptr, &m_compute.pipeline));

    // We can destroy the shader right away.
    shader.Cleanup();
    m_cleanupQueue.AddFunction([=] {
        vkDestroyPipelineLayout(m_device.logicalDevice, m_compute.pipelineLayout, nullptr);
        vkDestroyPipeline(m_device.logicalDevice, m_compute.pipeline, nullptr);
    });
}

void Application::RecordComputeCmd(VkCommandBuffer cmd) 
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compute.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_compute.pipelineLayout, 
        0, m_compute.dSets.size(), m_compute.dSets.data(), 
        0, nullptr);
    vkCmdDispatch(cmd, (m_image.extent.width / 16) + 1, (m_image.extent.height / 16) + 1, 1);   
}

void Application::RecordGraphicsCmd(VkCommandBuffer cmd, uint32_t swapchainImgIdx) 
{
    /*m_image.ChangeLayout(cmd,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, VK_ACCESS_SHADER_READ_BIT);*/

    // Begin renderpass
    
    VkClearValue clear;
    clear.color.float32[0] = 0.0f;
    clear.color.float32[1] = 0.0f;
    clear.color.float32[2] = 0.0f;
    clear.color.float32[3] = 1.0f;

    auto rpInfo = vkw::init::RenderPassBeginInfo(
        m_graphics.swapchain.renderPass, 
        m_graphics.swapchain.framebuffers[swapchainImgIdx], 
        m_graphics.swapchain.windowExtent,
        &clear);
    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphics.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_graphics.pipelineLayout, 
        0, m_graphics.dSets.size(), m_graphics.dSets.data(), 
        0, nullptr);
    // Draw the 6 vertices of the quad.
    vkCmdDraw(cmd, 6, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
}

void Application::SubmitGraphicsCmd(VkCommandBuffer cmd) 
{
    auto info = vkw::init::SubmitInfo(&cmd);

    VkSemaphore waitSemaphores[] = { 
        m_compute.semaphore, 
        m_graphics.imageReadySem };
    VkPipelineStageFlags waitMasks[] = { 
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    info.waitSemaphoreCount = 2;
    info.pWaitSemaphores = &waitSemaphores[0];
    info.pWaitDstStageMask = &waitMasks[0];

    VkSemaphore signalSemaphores[] = {
        m_graphics.semaphore,
        m_graphics.presentSem };
    info.signalSemaphoreCount = 2;
    info.pSignalSemaphores = &signalSemaphores[0];
    
    VK_CHECK(vkQueueSubmit(m_graphics.queue, 1, &info, m_graphics.fence));
}

void Application::SubmitComputeCmd(VkCommandBuffer cmd) 
{
    auto info = vkw::init::SubmitInfo(&cmd);

    VkPipelineStageFlags waitMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &m_graphics.semaphore;
    info.pWaitDstStageMask = &waitMask;

    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &m_compute.semaphore;

    VK_CHECK(vkQueueSubmit(m_compute.queue, 1, &info, m_compute.fence));
}

void Application::UpdateDDAUniforms() 
{
    DDAUniforms* contents = (DDAUniforms*)m_compute.ddaUniforms.Map();
    
    contents->screenResolution.x = m_image.extent.width;
    contents->screenResolution.y = m_image.extent.height;

    // Horizontal field of view in degrees.
    float FOVrad = glm::radians(m_compute.camera.fovDeg);
    contents->screenWorldSize.x = 2.0f * glm::tan(FOVrad / 2.0f);
    contents->screenWorldSize.y = contents->screenWorldSize.x * 
        (contents->screenResolution.y / (float)contents->screenResolution.x);
    
    // A dummy camera looking down the Z axis, with the Y axis facing up.
    contents->cameraPosition = glm::vec4(m_compute.camera.position, 0.0f);
    contents->cameraForward  = glm::vec4(m_compute.camera.forward, 0.0f);
    contents->cameraUp       = glm::vec4(m_compute.camera.Up(), 0.0f);
    contents->cameraRight    = glm::vec4(m_compute.camera.Right(), 0.0f);

    // Grid positions
    contents->gridWorldCoords = { -10.0f, -10.0f, -10.0f, 20.0f };
    contents->gridResolution = { m_compute.gridResolution, m_compute.gridResolution, m_compute.gridResolution, 0 };

    m_compute.ddaUniforms.Unmap(); 
}

void Application::UpdateDDAVoxels() 
{
    auto density = [] (float x, float y, float z) {
        return 32.0f*32.0f - (x*x+y*y+z*z);
    };

    DDAVoxel* contents = (DDAVoxel*)m_compute.ddaVoxels.Map();

    size_t res = m_compute.gridResolution;
    for (size_t x = 0; x < res; x++) {
        for (size_t y = 0; y < res; y++) {
            for (size_t z = 0; z < res; z++) {
                size_t index = z + y * res + x * res * res;
                float d = density(x, y, z);
                if (d >= 0.0f) {
                    float eps = 0.0001f;
                    contents[index].color = { 1.0f, 1.0f, 0.3f, 1.0f };
                    contents[index].normal = { 
                        (density(x + eps, y, z) - d) / eps,
                        (density(x, y + eps, z) - d) / eps,
                        (density(x, y, z + eps) - d) / eps,
                        0.0f };
                } 
                else {
                    contents[index].color = { 0.0f, 0.0f, 0.0f, 0.0f };
                    contents[index].normal = { 0.0f, 0.0f, 0.0f, 0.0f };
                }
            }
        }
    }
    m_compute.ddaVoxels.Unmap();    
}

void Application::UpdateDDALights() 
{
    void* contents = m_compute.ddaLights.Map();

    glm::uvec4* count = (glm::uvec4*)contents;
    count->x = m_compute.lightCount;

    DDALight* lights = (DDALight*)(reinterpret_cast<size_t>(contents) + sizeof(glm::uvec4));
    lights[0].direction = { -1, -1, 0, 0 };
    lights[0].color = { 1, 0, 0, 0 };

    lights[1].direction = { 1, -1, 0, 0 };
    lights[1].color = { 0, 0, 1, 0 };

    m_compute.ddaLights.Unmap();    
}

VkCommandBuffer Application::BuildCommand(
    VkCommandPool pool, 
    std::function<void(VkCommandBuffer)>&& record)
{
    // Reset the command pool (and its buffers).
    VK_CHECK(vkResetCommandPool(m_device.logicalDevice, pool, 0));

    // Allocate the command buffer.
    VkCommandBuffer cmd;
    auto allocInfo = vkw::init::CommandBufferAllocateInfo(pool);
    VK_CHECK(vkAllocateCommandBuffers(
        m_device.logicalDevice, &allocInfo, &cmd));

    // Begin the command.
    auto beginInfo = vkw::init::CommandBufferBeginInfo(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Record the command
    record(cmd);

    // End the command
    VK_CHECK(vkEndCommandBuffer(cmd));

    return cmd;
}

void Application::Draw() 
{
    m_compute.camera.Update(m_inputManager);

    // Compute command.
    // Wait for the previous command to finish.
    VK_CHECK(vkWaitForFences(m_device.logicalDevice, 1, &m_compute.fence, true, 1000000000));
    VK_CHECK(vkResetFences(m_device.logicalDevice, 1, &m_compute.fence));
    // Update the uniform buffer
    UpdateDDAUniforms();
    // Submit a new command
    auto computeCmd = BuildCommand(m_compute.cmdPool, 
        [=] (VkCommandBuffer cmd) { RecordComputeCmd(cmd); });
    SubmitComputeCmd(computeCmd);
    
    // Acquire image.
    uint32_t swapchainImgIdx = m_graphics.swapchain.AcquireNewImage(
        m_graphics.imageReadySem);

    // Render command.
    // Wait for the previous command to finish.
    VK_CHECK(vkWaitForFences(m_device.logicalDevice, 1, &m_graphics.fence, true, 1000000000));
    VK_CHECK(vkResetFences(m_device.logicalDevice, 1, &m_graphics.fence));
    auto graphicsCmd = BuildCommand(m_graphics.cmdPool,
        [=] (VkCommandBuffer cmd) { RecordGraphicsCmd(cmd, swapchainImgIdx); });
    SubmitGraphicsCmd(graphicsCmd);

    // Present image.
    auto presentInfo = vkw::init::PresentInfoKHR(
        &m_graphics.swapchain.swapchain, &swapchainImgIdx);
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &m_graphics.presentSem;
    VK_CHECK(vkQueuePresentKHR(m_graphics.queue, &presentInfo));  
}

