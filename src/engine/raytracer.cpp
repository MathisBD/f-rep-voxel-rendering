#include "engine/raytracer.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/shader.h"
#include <glm/gtx/norm.hpp>
#include "utils/timer.h"


void Raytracer::Init(
    vkw::Device* device, 
    vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
    RenderTarget* target, VoxelStorage* voxels) 
{
    m_device = device;
    m_target = target;
    m_descAllocator = descAllocator;
    m_descCache = descCache;
    m_voxels = voxels;

    InitCommands();
    InitSynchronization();
    InitBuffers();
    InitPipeline();
    InitUploadCtxt();
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
    m_paramsBuffer.Init(m_device);
    m_paramsBuffer.Allocate(
        sizeof(ShaderParams), 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_device->NameObject(m_paramsBuffer.buffer, "raytracer params buffer");
    m_cleanupQueue.AddFunction([=] { m_paramsBuffer.Cleanup(); });
}

void Raytracer::InitPipeline() 
{
    // Load the shader
    vkw::Shader shader;
    shader.Init(m_device->logicalDevice, "../shaders/raycaster/main.comp.spv");
    
    // Descriptor Sets
    m_descSets = std::vector<VkDescriptorSet>(1);
    auto dSetLayouts = std::vector<VkDescriptorSetLayout>(1);

    // Descriptor Set 0    
    auto outImageInfo = vkw::init::DescriptorImageInfo(
        m_target->sampler, m_target->view, VK_IMAGE_LAYOUT_GENERAL);
    auto uniformsInfo = vkw::init::DescriptorBufferInfo(
        m_paramsBuffer.buffer, 0, m_paramsBuffer.size);
    auto nodeInfo = vkw::init::DescriptorBufferInfo(
        m_voxels->nodeBuffer.buffer, 0, m_voxels->nodeBuffer.size);
    auto tapeInfo = vkw::init::DescriptorBufferInfo(
        m_voxels->tapeBuffer.buffer, 0, m_voxels->tapeBuffer.size);
    vkw::DescriptorBuilder(m_descCache, m_descAllocator)
        .BindImage( 0, &outImageInfo,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(1, &uniformsInfo,  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(2, &nodeInfo,      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(3, &tapeInfo,      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .Build(&m_descSets[0], &dSetLayouts[0]);
    m_device->NameObject(m_descSets[0], "raytracer descriptor set 0");

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
    m_device->NameObject(m_pipeline, "raytracer pipeline");
    
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


void Raytracer::UpdateShaderParams(const Camera* camera, float tapeTime) 
{
    assert(m_voxels->gridLevels < MAX_LEVEL_COUNT);
    ShaderParams* params = (ShaderParams*)m_paramsBuffer.Map();
    
    params->lightCount = 2;
    params->levelCount = m_voxels->gridLevels;
    params->tapeInstrCount = m_voxels->tape.instructions.size(); 
    params->tapeTime = tapeTime;

    params->outImgLayer = m_targetImgLayer;
    m_targetImgLayer = (m_targetImgLayer + 1) % m_target->temporalSampleCount;
    params->time = Timer::s_time;
    
    // Grid positions
    params->gridWorldCoords = m_voxels->lowVertex;
    params->gridWorldSize = m_voxels->worldSize;
    
    params->screenResolution.x = m_target->image.extent.width;
    params->screenResolution.y = m_target->image.extent.height;

    // Horizontal field of view in degrees.
    float FOVrad = glm::radians(camera->fovDeg);
    params->screenWorldSize.x = 2.0f * glm::tan(FOVrad / 2.0f);
    params->screenWorldSize.y = params->screenWorldSize.x * 
        (params->screenResolution.y / (float)params->screenResolution.x);
    
    // A dummy camera looking down the Z axis, with the Y axis facing up.
    params->cameraPosition = glm::vec4(camera->position, 0.0f);
    params->cameraForward  = glm::vec4(camera->forward, 0.0f);
    params->cameraUp       = glm::vec4(camera->Up(), 0.0f);
    params->cameraRight    = glm::vec4(camera->Right(), 0.0f);

    // Level data
    uint32_t nodeOfs = 0;
    float cellSize = m_voxels->worldSize;
    for (uint32_t i = 0; i < m_voxels->gridDims.size(); i++) {
        cellSize /= m_voxels->gridDims[i];

        params->levels[i].dim = m_voxels->gridDims[i];
        params->levels[i].nodeOfs = nodeOfs;
        params->levels[i].cellSize = cellSize;

        // Remember : the shader-side offsets are in uints (not in bytes).
        nodeOfs += m_voxels->interiorNodeCount[i] * m_voxels->NodeSize(i) / sizeof(uint32_t);
    }
    
    /*for (uint32_t i = 0; i < m_voxels->gridDims.size(); i++) {
        printf("\n");
        printf("node count[%u] = %u\n", i, m_voxels->nodeCount[i]);
        printf("node ofs[%u] = %u\n", i, params->levels[i].nodeOfs);
        printf("child ofs[%u] = %u\n", i, params->levels[i].childOfs);
        printf("node size(%u)=%u\n", i, m_voxels->NodeSize(i));
    }*/

    // Background color
    params->backgroundColor = glm::vec4(m_backgroundColor, 1.0f);

    // Lights
    params->lights[0].direction = glm::normalize(glm::vec4({ -1, -1, -0.2, 0 }));
    params->lights[0].color = { 1, 0, 0, 0 };

    params->lights[1].direction = glm::normalize(glm::vec4({ 1, -1, -0.2, 0 }));
    params->lights[1].color = { 0, 0, 2, 0 };

    // Tape constant pool
    assert(m_voxels->tape.constantPool.size() <= MAX_CONSTANT_POOL_SIZE);
    memcpy(params->constantPool, m_voxels->tape.constantPool.data(), 
        m_voxels->tape.constantPool.size() * sizeof(float));

    m_paramsBuffer.Unmap(); 
}


void Raytracer::RecordComputeCmd(VkCommandBuffer cmd) 
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 
        0, m_descSets.size(), m_descSets.data(), 
        0, nullptr);
    vkCmdDispatch(cmd, 
        (m_target->image.extent.width / THREAD_GROUP_SIZE_X) + 1, 
        (m_target->image.extent.height / THREAD_GROUP_SIZE_Y) + 1, 
        1);   
}

void Raytracer::SubmitComputeCmd(VkCommandBuffer cmd, VkSemaphore waitSem) 
{
    auto info = vkw::init::SubmitInfo(&cmd);

    VkPipelineStageFlags waitDstMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &waitSem;
    info.pWaitDstStageMask = &waitDstMask;

    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &m_semaphore;

    VK_CHECK(vkQueueSubmit(m_queue, 1, &info, m_fence));
}

void Raytracer::Trace(VkSemaphore waitSem, const Camera* camera, float time) 
{
    // Wait for the previous command to finish.
    VK_CHECK(vkWaitForFences(m_device->logicalDevice, 1, &m_fence, true, 1000000000));
    VK_CHECK(vkResetFences(m_device->logicalDevice, 1, &m_fence));
    
    // Update the uniform buffer
    UpdateShaderParams(camera, time);

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
    SubmitComputeCmd(cmd, waitSem);
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


uint32_t Raytracer::SplitBy3(uint32_t x) 
{
    x &= 0xFF;                     // 0000 0000 0000 0000 1111 1111
    x = (x | (x << 8)) & 0x00F00F; // 0000 0000 1111 0000 0000 1111
    x = (x | (x << 4)) & 0x0c30c3; // 0000 1100 0011 0000 1100 0011
    x = (x | (x << 2)) & 0x249249; // 0010 0100 1001 0010 0100 1001
    return x;
}

uint32_t Raytracer::MortonEncode(glm::u32vec3 cell) 
{
    return SplitBy3(cell.x) | (SplitBy3(cell.y) << 1) | (SplitBy3(cell.z) << 2);
}