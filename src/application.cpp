#include "application.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/shader.h"
#include "vk_wrapper/pipeline_builder.h"
#include <glm/gtx/norm.hpp>


void Application::Init()
{
    EngineBase::Init();

    printf("[+] Queue families :\n\tgraphics=%u\n\tcompute=%u\n\ttransfer=%u\n",
        m_device.queueFamilies.graphics, 
        m_device.queueFamilies.compute,
        m_device.queueFamilies.transfer);

    InitRenderTarget();

    m_renderer.Init(
        &m_device, &m_descriptorAllocator, &m_descriptorCache,
        &m_target, m_windowExtent, m_surface);
    m_cleanupQueue.AddFunction([=] { m_renderer.Cleanup(); });

    InitCompute();
    InitComputePipeline();
}

void Application::InitRenderTarget() 
{
    // Allocate the image
    m_target.image.Init(m_vmaAllocator);
    std::vector<uint32_t> queueFamilies;
    VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (m_device.queueFamilies.graphics != m_device.queueFamilies.compute) {
        queueFamilies = { 
            m_device.queueFamilies.graphics,
            m_device.queueFamilies.compute };
        sharingMode = VK_SHARING_MODE_CONCURRENT;
    }
    m_target.image.Allocate(
        m_windowExtent,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        sharingMode,
        &queueFamilies);
    m_cleanupQueue.AddFunction([=] { m_target.image.Cleanup(); });

    // Set the image layout (GENERAL).
    ImmediateSubmit(
        [=] (VkCommandBuffer cmd) { 
            m_target.image.ChangeLayout(cmd, 
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, VK_ACCESS_SHADER_WRITE_BIT);
        });

    // Image view
    auto viewInfo = vkw::init::ImageViewCreateInfo(
        m_target.image.format, m_target.image.image, VK_IMAGE_ASPECT_COLOR_BIT);
    VK_CHECK(vkCreateImageView(m_device.logicalDevice, &viewInfo, nullptr, &m_target.view));
    m_cleanupQueue.AddFunction([=] {
        vkDestroyImageView(m_device.logicalDevice, m_target.view, nullptr);
    });

    // Sampler
    auto samplerInfo = vkw::init::SamplerCreateInfo(VK_FILTER_NEAREST);
    VK_CHECK(vkCreateSampler(
        m_device.logicalDevice, &samplerInfo, nullptr, &m_target.sampler));
    m_cleanupQueue.AddFunction([=] {
        vkDestroySampler(m_device.logicalDevice, m_target.sampler, nullptr);
    });
}

void Application::RecordComputeCmd(VkCommandBuffer cmd) 
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compute.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_compute.pipelineLayout, 
        0, m_compute.dSets.size(), m_compute.dSets.data(), 
        0, nullptr);
    vkCmdDispatch(cmd, (m_target.image.extent.width / 16) + 1, (m_target.image.extent.height / 16) + 1, 1);   
}

void Application::SubmitComputeCmd(VkCommandBuffer cmd) 
{
    auto info = vkw::init::SubmitInfo(&cmd);

    VkSemaphore waitSem = m_renderer.GetRenderSemaphore();
    VkPipelineStageFlags waitMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &waitSem;
    info.pWaitDstStageMask = &waitMask;

    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &m_compute.semaphore;

    VK_CHECK(vkQueueSubmit(m_compute.queue, 1, &info, m_compute.fence));
}

void Application::UpdateDDAUniforms() 
{
    DDAUniforms* contents = (DDAUniforms*)m_compute.ddaUniforms.Map();
    
    contents->screenResolution.x = m_target.image.extent.width;
    contents->screenResolution.y = m_target.image.extent.height;

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
    contents->gridWorldCoords = glm::vec4(m_compute.voxelGrid.lowVertex, m_compute.voxelGrid.worldSize);
    contents->gridResolution = { m_compute.voxelGrid.dim, m_compute.voxelGrid.dim, m_compute.voxelGrid.dim, 0 };

    m_compute.ddaUniforms.Unmap(); 
}

void Application::UpdateDDAVoxels() 
{
    auto density = [] (float x, float y, float z) {
        glm::vec3 pos = { x, y, z };
        glm::vec3 center = { 0, 0, 0 };
        float radius = 5;
        return radius*radius - glm::length2(pos - center);
    };

    DDAVoxel* contents = (DDAVoxel*)m_compute.ddaVoxels.Map();

    size_t dim = m_compute.voxelGrid.dim;
    for (size_t x = 0; x < dim; x++) {
        for (size_t y = 0; y < dim; y++) {
            for (size_t z = 0; z < dim; z++) {
                size_t index = z + y * dim + x * dim * dim;
                glm::vec3 pos = m_compute.voxelGrid.WorldPosition({ x, y, z });
                float d = density(pos.x, pos.y, pos.z);
                if (d >= 0.0f) {
                    float eps = 0.001f;
                    contents[index].color = { 1.0f, 1.0f, 0.3f, 1.0f };
                    contents[index].normal = { 
                        (density(pos.x + eps, pos.y, pos.z) - d) / eps,
                        (density(pos.x, pos.y + eps, pos.z) - d) / eps,
                        (density(pos.x, pos.y, pos.z + eps) - d) / eps,
                        0.0f };
                    contents[index].normal = -glm::normalize(contents[index].normal);
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
    lights[1].color = { 0, 0, 2, 0 };

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
    // Update the uniform buffers
    UpdateDDAUniforms();
    UpdateDDALights();
    // Submit a new command
    auto computeCmd = BuildCommand(m_compute.cmdPool, 
        [=] (VkCommandBuffer cmd) { RecordComputeCmd(cmd); });
    SubmitComputeCmd(computeCmd);
    
    m_renderer.BeginFrame();
    m_renderer.Render(m_compute.semaphore);
    m_renderer.EndFrame();
}

