#include "engine/raytracer.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/shader.h"
#include <glm/gtx/norm.hpp>



void Raytracer::Init(
    vkw::Device* device, 
    vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
    RenderTarget* target, VmaAllocator vmaAllocator) 
{
    m_device = device;
    m_target = target;
    m_vmaAllocator = vmaAllocator;
    m_descAllocator = descAllocator;
    m_descCache = descCache;

    // Voxel grid
    m_voxelGrid = CubeGrid(512, { -10, -10, -10 }, 20);

    InitCommands();
    InitSynchronization();
    InitBuffers();
    InitPipeline();
    InitUploadCtxt();

    // We only upload the voxels once.    
    auto density = [] (float x, float y, float z) {
        glm::vec3 pos = { x, y, z };
        glm::vec3 center = { 0, 0, 0 };
        float radius = 5;
        return radius * radius - glm::length2(pos - center);
    };
    UpdateVoxelBuffers(density);
}

void Raytracer::Cleanup() 
{
    m_cleanupQueue.Flush();    
}


void Raytracer::InitCommands() 
{
    // Queue
    vkGetDeviceQueue(m_device->logicalDevice, 
        m_device->queueFamilies.compute, 0, &m_queue);
    
    // Command pool
    auto poolInfo = vkw::init::CommandPoolCreateInfo(
        m_device->queueFamilies.compute);
    VK_CHECK(vkCreateCommandPool(m_device->logicalDevice, &poolInfo, nullptr, &m_cmdPool));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyCommandPool(m_device->logicalDevice, m_cmdPool, nullptr); 
    });
}

void Raytracer::InitSynchronization()
{
    // Fence
    auto fenceInfo = vkw::init::FenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(m_device->logicalDevice, &fenceInfo, nullptr, &m_fence));
    m_cleanupQueue.AddFunction([=] {
        vkDestroyFence(m_device->logicalDevice, m_fence, nullptr);
    });

    // Semaphore    
    auto semInfo = vkw::init::SemaphoreCreateInfo();
    VK_CHECK(vkCreateSemaphore(m_device->logicalDevice, &semInfo, nullptr, &m_semaphore));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroySemaphore(m_device->logicalDevice, m_semaphore, nullptr);
    });
}

void Raytracer::InitBuffers()
{
    // Uniform buffer
    m_buffers.uniforms.Init(m_vmaAllocator);
    m_buffers.uniforms.Allocate(
        sizeof(DDAUniforms), 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] { m_buffers.uniforms.Cleanup(); });

    // Voxel data buffer
    size_t voxelCount = m_voxelGrid.dim * m_voxelGrid.dim * m_voxelGrid.dim;
    m_buffers.voxelData.Init(m_vmaAllocator);
    m_buffers.voxelData.Allocate(
        voxelCount * sizeof(DDAVoxel), 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    m_cleanupQueue.AddFunction([=] { m_buffers.voxelData.Cleanup(); });

    // Voxel mask buffer
    size_t bitsPerByte = 8;
    m_buffers.voxelMask.Init(m_vmaAllocator);
    m_buffers.voxelMask.Allocate(
        voxelCount / bitsPerByte, // there are voxelCount bits in the mask.
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);
    m_cleanupQueue.AddFunction([=] { m_buffers.voxelMask.Cleanup(); });
}

void Raytracer::InitPipeline() 
{
    // Load the shader
    vkw::Shader shader;
    shader.Init(m_device->logicalDevice, "../shaders/dda.comp.spv");
    
    // Descriptor Sets
    m_descSets = std::vector<VkDescriptorSet>(1);
    auto dSetLayouts = std::vector<VkDescriptorSetLayout>(1);

    // Descriptor Set 0    
    auto outImageInfo = vkw::init::DescriptorImageInfo(
        m_target->sampler, m_target->view, VK_IMAGE_LAYOUT_GENERAL);
    auto uniformsInfo = vkw::init::DescriptorBufferInfo(
        m_buffers.uniforms.buffer, 0, m_buffers.uniforms.size);
    auto voxelDataInfo = vkw::init::DescriptorBufferInfo(
        m_buffers.voxelData.buffer, 0, m_buffers.voxelData.size);
    auto voxelMaskInfo = vkw::init::DescriptorBufferInfo(
        m_buffers.voxelMask.buffer, 0, m_buffers.voxelMask.size);
    vkw::DescriptorBuilder(m_descCache, m_descAllocator)
        .BindImage(0, &outImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(1, &uniformsInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(2, &voxelDataInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(3, &voxelMaskInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .Build(&m_descSets[0], &dSetLayouts[0]);

    // Pipeline layout
    auto layoutInfo = vkw::init::PipelineLayoutCreateInfo();
    layoutInfo.setLayoutCount = (uint32_t)dSetLayouts.size();
    layoutInfo.pSetLayouts = dSetLayouts.data();
    VK_CHECK(vkCreatePipelineLayout(m_device->logicalDevice, &layoutInfo, nullptr, &m_pipelineLayout));
 
    // Pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = nullptr;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.stage = vkw::init::PipelineShaderStageCreateInfo(
        VK_SHADER_STAGE_COMPUTE_BIT, shader.shader);
    VK_CHECK(vkCreateComputePipelines(m_device->logicalDevice, VK_NULL_HANDLE, 
        1, &pipelineInfo, nullptr, &m_pipeline));

    // We can destroy the shader right away.
    shader.Cleanup();
    m_cleanupQueue.AddFunction([=] {
        vkDestroyPipelineLayout(m_device->logicalDevice, m_pipelineLayout, nullptr);
        vkDestroyPipeline(m_device->logicalDevice, m_pipeline, nullptr);
    });
}

void Raytracer::InitUploadCtxt() 
{
    // Command pool
    auto poolInfo = vkw::init::CommandPoolCreateInfo(
        m_device->queueFamilies.compute);
    VK_CHECK(vkCreateCommandPool(m_device->logicalDevice, &poolInfo, nullptr, &m_uploadCtxt.cmdPool));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyCommandPool(m_device->logicalDevice, m_uploadCtxt.cmdPool, nullptr); 
    });

    // Fence
    auto fenceInfo = vkw::init::FenceCreateInfo();
    VK_CHECK(vkCreateFence(m_device->logicalDevice, &fenceInfo, nullptr, &m_uploadCtxt.fence));
    m_cleanupQueue.AddFunction([=] {
        vkDestroyFence(m_device->logicalDevice, m_uploadCtxt.fence, nullptr);
    });
}


void Raytracer::UpdateUniformBuffer(const Camera* camera) 
{
    DDAUniforms* contents = (DDAUniforms*)m_buffers.uniforms.Map();
    
    contents->screenResolution.x = m_target->image.extent.width;
    contents->screenResolution.y = m_target->image.extent.height;

    // Horizontal field of view in degrees.
    float FOVrad = glm::radians(camera->fovDeg);
    contents->screenWorldSize.x = 2.0f * glm::tan(FOVrad / 2.0f);
    contents->screenWorldSize.y = contents->screenWorldSize.x * 
        (contents->screenResolution.y / (float)contents->screenResolution.x);
    
    // A dummy camera looking down the Z axis, with the Y axis facing up.
    contents->cameraPosition = glm::vec4(camera->position, 0.0f);
    contents->cameraForward  = glm::vec4(camera->forward, 0.0f);
    contents->cameraUp       = glm::vec4(camera->Up(), 0.0f);
    contents->cameraRight    = glm::vec4(camera->Right(), 0.0f);

    // Grid positions
    contents->gridWorldCoords = glm::vec4(m_voxelGrid.lowVertex, m_voxelGrid.worldSize);
    contents->gridResolution = { m_voxelGrid.dim, 0, 0, 0 };

    // Background color
    contents->backgroundColor = glm::vec4(m_backgroundColor, 1.0f);

    // Lights
    m_lightCount = 2;
    contents->lightCount = { m_lightCount, 0, 0, 0 };

    contents->lights[0].direction = { -1, -1, 0, 0 };
    contents->lights[0].color = { 1, 0, 0, 0 };

    contents->lights[1].direction = { 1, -1, 0, 0 };
    contents->lights[1].color = { 0, 0, 2, 0 };

    m_buffers.uniforms.Unmap(); 
}

void Raytracer::UpdateVoxelBuffers(std::function<float(float, float, float)>&& density) 
{
    vkw::Buffer stagingMask;
    stagingMask.Init(m_vmaAllocator);
    stagingMask.Allocate(m_buffers.voxelMask.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    vkw::Buffer stagingData;
    stagingData.Init(m_vmaAllocator);
    stagingData.Allocate(m_buffers.voxelData.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    uint32_t* mask = (uint32_t*)stagingMask.Map();
    DDAVoxel* data = (DDAVoxel*)stagingData.Map();
    memset(mask, 0, stagingMask.size);
    memset(data, 0, stagingData.size);
    
    size_t dim = m_voxelGrid.dim;
    for (size_t x = 0; x < dim; x++) {
        for (size_t y = 0; y < dim; y++) {
            for (size_t z = 0; z < dim; z++) {
                size_t index = z + y * dim + x * dim * dim;
                glm::vec3 pos = m_voxelGrid.WorldPosition({ x, y, z });
                float d = density(pos.x, pos.y, pos.z);

                if (d >= 0.0f) {
                    mask[index >> 5] |= (1 << (index & (0x1F)));
                    
                    float eps = 0.001f;
                    data[index].color = { 1.0f, 1.0f, 0.3f, 1.0f };
                    data[index].normal = { 
                        (density(pos.x + eps, pos.y, pos.z) - d) / eps,
                        (density(pos.x, pos.y + eps, pos.z) - d) / eps,
                        (density(pos.x, pos.y, pos.z + eps) - d) / eps,
                        0.0f };
                    data[index].normal = -glm::normalize(data[index].normal);
                }
            }
        }
    }
    stagingData.Unmap();
    stagingMask.Unmap();

    // Copy the data to the GPU.
    ImmediateSubmit([=] (VkCommandBuffer cmd) { 
        VkBufferCopy maskRegion;
        maskRegion.srcOffset = 0;
        maskRegion.dstOffset = 0;
        maskRegion.size = m_buffers.voxelMask.size;
        vkCmdCopyBuffer(cmd, stagingMask.buffer, m_buffers.voxelMask.buffer, 1, &maskRegion);
    
        VkBufferCopy dataRegion;
        dataRegion.srcOffset = 0;
        dataRegion.dstOffset = 0;
        dataRegion.size = m_buffers.voxelData.size;
        vkCmdCopyBuffer(cmd, stagingData.buffer, m_buffers.voxelData.buffer, 1, &dataRegion);
    });
    stagingMask.Cleanup();
    stagingData.Cleanup();
}


void Raytracer::RecordComputeCmd(VkCommandBuffer cmd) 
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 
        0, m_descSets.size(), m_descSets.data(), 
        0, nullptr);
    vkCmdDispatch(cmd, (m_target->image.extent.width / 16) + 1, (m_target->image.extent.height / 16) + 1, 1);   
}

void Raytracer::SubmitComputeCmd(VkCommandBuffer cmd, VkSemaphore renderSem) 
{
    auto info = vkw::init::SubmitInfo(&cmd);

    VkPipelineStageFlags waitMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &renderSem;
    info.pWaitDstStageMask = &waitMask;

    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &m_semaphore;

    VK_CHECK(vkQueueSubmit(m_queue, 1, &info, m_fence));
}

void Raytracer::Trace(VkSemaphore renderSem, const Camera* camera) 
{
    // Wait for the previous command to finish.
    VK_CHECK(vkWaitForFences(m_device->logicalDevice, 1, &m_fence, true, 1000000000));
    VK_CHECK(vkResetFences(m_device->logicalDevice, 1, &m_fence));
    
    // Update the uniform buffer
    UpdateUniformBuffer(camera);

    // Reset the command pool (and its buffers).
    VK_CHECK(vkResetCommandPool(m_device->logicalDevice, m_cmdPool, 0));
    // Allocate the command buffer.
    VkCommandBuffer cmd;
    auto allocInfo = vkw::init::CommandBufferAllocateInfo(m_cmdPool);
    VK_CHECK(vkAllocateCommandBuffers(m_device->logicalDevice, &allocInfo, &cmd));
    // Begin the command.
    auto beginInfo = vkw::init::CommandBufferBeginInfo(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
    // Record the command
    RecordComputeCmd(cmd);
    // End the command
    VK_CHECK(vkEndCommandBuffer(cmd));
    // Submit.
    SubmitComputeCmd(cmd, renderSem);
}

void Raytracer::SetBackgroundColor(const glm::vec3& color) 
{
    m_backgroundColor = color;    
}

void Raytracer::ImmediateSubmit(std::function<void(VkCommandBuffer)>&& record) 
{ 
    VkCommandBuffer cmd;
    auto allocInfo = vkw::init::CommandBufferAllocateInfo(m_uploadCtxt.cmdPool);
    VK_CHECK(vkAllocateCommandBuffers(
        m_device->logicalDevice, &allocInfo, &cmd));

    auto beginInfo = vkw::init::CommandBufferBeginInfo(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    record(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    auto submitInfo = vkw::init::SubmitInfo(&cmd);
    VK_CHECK(vkQueueSubmit(
        m_queue, 1, &submitInfo, m_uploadCtxt.fence));

    // Wait for the command to finish
    VK_CHECK(vkWaitForFences(
        m_device->logicalDevice, 1, &m_uploadCtxt.fence, VK_TRUE, 1000000000));
    VK_CHECK(vkResetFences(
        m_device->logicalDevice, 1, &m_uploadCtxt.fence));

    VK_CHECK(vkResetCommandPool(
        m_device->logicalDevice, m_uploadCtxt.cmdPool, 0));
}