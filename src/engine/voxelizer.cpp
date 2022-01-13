#include "engine/voxelizer.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/shader.h"
#include <cstring>


void Voxelizer::Init(
    vkw::Device* device, 
    vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
    VoxelStorage* voxels, VmaAllocator vmaAllocator) 
{
    m_device = device;
    m_descAllocator = descAllocator;
    m_descCache = descCache;
    m_voxels = voxels;
    m_vmaAllocator = vmaAllocator;

    InitCommands();
    InitSynchronization();
    InitBuffers();

    AllocateGPUBuffers();
    ZeroOutNodeBuffer();
    UploadTape();

    InitPipeline();
}

void Voxelizer::Cleanup() 
{
    m_cleanupQueue.Flush();
}

void Voxelizer::InitCommands() 
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

void Voxelizer::InitSynchronization() 
{
    // Fence
    auto fenceInfo = vkw::init::FenceCreateInfo();
    VK_CHECK(vkCreateFence(m_device->logicalDevice, &fenceInfo, nullptr, &m_fence));
    m_cleanupQueue.AddFunction([=] {
        vkDestroyFence(m_device->logicalDevice, m_fence, nullptr);
    });

    // Semaphores
    m_voxelizeLevelSems = std::vector<VkSemaphore>(m_voxels->gridLevels);
    auto semInfo = vkw::init::SemaphoreCreateInfo();
    for (uint32_t i = 0; i < m_voxels->gridLevels; i++) {
        VK_CHECK(vkCreateSemaphore(m_device->logicalDevice, 
            &semInfo, nullptr, &m_voxelizeLevelSems[i]));
        m_cleanupQueue.AddFunction([=] {
            vkDestroySemaphore(m_device->logicalDevice, m_voxelizeLevelSems[i], nullptr);
        });
    }
}

void Voxelizer::InitBuffers() 
{
    // Params buffer
    m_paramsBuffer.Init(m_vmaAllocator);
    m_paramsBuffer.Allocate(
        sizeof(ShaderParams), 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] { m_paramsBuffer.Cleanup(); });

    // Child count buffer
    m_childCountBuffer.Init(m_vmaAllocator);
    m_childCountBuffer.Allocate(
        sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] { m_childCountBuffer.Cleanup(); });
}

void Voxelizer::InitPipeline() 
{
    // Load the shader
    vkw::Shader shader;
    shader.Init(m_device->logicalDevice, "../shaders/voxelizer/main.comp.spv");

    // Descriptor Sets
    m_descSets = std::vector<VkDescriptorSet>(1);
    auto dSetLayouts = std::vector<VkDescriptorSetLayout>(1);

    // Descriptor Set 0.
    auto paramsInfo = vkw::init::DescriptorBufferInfo(
        m_paramsBuffer.buffer, 0, m_paramsBuffer.size);
    auto nodeInfo = vkw::init::DescriptorBufferInfo(
        m_voxels->nodeBuffer.buffer, 0, m_voxels->nodeBuffer.size);
    auto tapeInfo = vkw::init::DescriptorBufferInfo(
        m_voxels->tapeBuffer.buffer, 0, m_voxels->tapeBuffer.size);
    auto childCountInfo = vkw::init::DescriptorBufferInfo(
        m_childCountBuffer.buffer, 0, m_childCountBuffer.size);
    vkw::DescriptorBuilder(m_descCache, m_descAllocator)
        .BindBuffer(0, &paramsInfo,     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(1, &nodeInfo,       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(2, &tapeInfo,       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(3, &childCountInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
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

void Voxelizer::UpdateShaderParams(uint32_t level, float time) 
{
    ShaderParams* params = (ShaderParams*)m_paramsBuffer.Map();
    params->levelCount = m_voxels->gridLevels;
    params->level = level;
    params->tapeInstrCount = m_voxels->tape.instructions.size();
    params->time = time;

    params->gridWorldCoords = m_voxels->lowVertex;
    params->gridWorldSize = m_voxels->worldSize;

    uint32_t nodeOfs = 0;
    float cellSize = m_voxels->worldSize;
    for (uint32_t i = 0; i <= level+1 && i < m_voxels->gridLevels; i++) {
        cellSize /= m_voxels->gridDims[i];

        params->levels[i].dim = m_voxels->gridDims[i];
        params->levels[i].cellSize = cellSize;
        params->levels[i].nodeOfs = nodeOfs;

        // Remember : the shader-side offsets are in uints (not in bytes).
        nodeOfs += m_voxels->interiorNodeCount[i] * m_voxels->NodeSize(i) / sizeof(uint32_t);
    }
    //params->levels[0].nodeOfs = 0;
    //params->levels[1].nodeOfs = m_voxels->NodeSize(0) / sizeof(uint32_t);

    memcpy(params->constantPool, m_voxels->tape.constantPool.data(),
        m_voxels->tape.constantPool.size() * sizeof(csg::Tape::Instr));

    m_paramsBuffer.Unmap();

    uint32_t* childCount = (uint32_t*)m_childCountBuffer.Map();
    *childCount = 0;
    m_childCountBuffer.Unmap();
}

void Voxelizer::RecordCmd(VkCommandBuffer cmd, uint32_t level) 
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 
        0, m_descSets.size(), m_descSets.data(), 
        0, nullptr);
    //vkCmdDispatch(cmd, m_voxels->interiorNodeCount[level], 1, 1);   
    uint32_t dim = m_voxels->gridDims[level];
    assert(dim % THREAD_GROUP_SIZE_X == 0);
    assert(dim % THREAD_GROUP_SIZE_Y == 0);
    assert(dim % THREAD_GROUP_SIZE_Z == 0);
    vkCmdDispatch(cmd, 
        (dim / THREAD_GROUP_SIZE_X) * m_voxels->interiorNodeCount[level], 
        dim / THREAD_GROUP_SIZE_Y, 
        dim / THREAD_GROUP_SIZE_Z);
}

void Voxelizer::SubmitCmd(VkCommandBuffer cmd, uint32_t level, VkSemaphore waitSem) 
{
    auto info = vkw::init::SubmitInfo(&cmd);

    VkPipelineStageFlags waitMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    info.waitSemaphoreCount = 1;
    if (level == 0) {
        info.pWaitSemaphores = &waitSem;
    }
    else {
        info.pWaitSemaphores = &m_voxelizeLevelSems[level - 1];
    }
    info.pWaitDstStageMask = &waitMask;
        
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &m_voxelizeLevelSems[level];

    VK_CHECK(vkQueueSubmit(m_queue, 1, &info, m_fence));
}

void Voxelizer::VoxelizeLevel(uint32_t level, VkSemaphore waitSem, float time) 
{
    // Update the uniform buffer
    UpdateShaderParams(level, time);

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
    RecordCmd(cmd, level);
    // End the command
    VK_CHECK(vkEndCommandBuffer(cmd));
    // Submit.
    SubmitCmd(cmd, level, waitSem);
    
    // Wait for the command to finish.
    VK_CHECK(vkWaitForFences(m_device->logicalDevice, 1, &m_fence, true, 1000000000));
    VK_CHECK(vkResetFences(m_device->logicalDevice, 1, &m_fence));

    uint32_t* childCount = (uint32_t*)m_childCountBuffer.Map();
    m_voxels->interiorNodeCount[level + 1] = *childCount;
    m_childCountBuffer.Unmap();
}

void Voxelizer::AllocateGPUBuffers() 
{
    // Start with 1GB
    uint32_t nodeSize = 1 << 30;
    uint32_t tapeSize = m_voxels->tape.instructions.size() * sizeof(csg::Tape::Instr);

    m_voxels->nodeBuffer.Allocate(nodeSize,   
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VMA_MEMORY_USAGE_GPU_ONLY);
    m_voxels->tapeBuffer.Allocate(tapeSize,   
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VMA_MEMORY_USAGE_GPU_ONLY);
}

void Voxelizer::ZeroOutNodeBuffer() 
{
    // Staging buffer
    vkw::Buffer nodeStagingBuf;
    nodeStagingBuf.Init(m_vmaAllocator);
    nodeStagingBuf.Allocate(m_voxels->nodeBuffer.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* nodeContents = nodeStagingBuf.Map();
    memset(nodeContents, 0, nodeStagingBuf.size);
    nodeStagingBuf.Unmap();

    // Allocate the command buffer.
    VkCommandBuffer cmd;
    auto allocInfo = vkw::init::CommandBufferAllocateInfo(m_cmdPool);
    VK_CHECK(vkAllocateCommandBuffers(m_device->logicalDevice, &allocInfo, &cmd));
    // Begin the command.
    auto beginInfo = vkw::init::CommandBufferBeginInfo(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
    
    // Record the command 
    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = nodeStagingBuf.size;
    vkCmdCopyBuffer(cmd, nodeStagingBuf.buffer, m_voxels->nodeBuffer.buffer, 1, &region);
    
    // End the command
    VK_CHECK(vkEndCommandBuffer(cmd));
    // Submit.
    auto submitInfo = vkw::init::SubmitInfo(&cmd);
    VK_CHECK(vkQueueSubmit(m_queue, 1, &submitInfo, m_fence));

    // Wait for the command to finish.
    VK_CHECK(vkWaitForFences(m_device->logicalDevice, 1, &m_fence, true, 1000000000));
    VK_CHECK(vkResetFences(m_device->logicalDevice, 1, &m_fence));

    nodeStagingBuf.Cleanup();
}

void Voxelizer::UploadTape() 
{
    vkw::Buffer tapeStagingBuf;
    tapeStagingBuf.Init(m_vmaAllocator);
    tapeStagingBuf.Allocate(m_voxels->tapeBuffer.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    void* tapeContents = tapeStagingBuf.Map();
    memcpy(tapeContents, m_voxels->tape.instructions.data(),
        m_voxels->tape.instructions.size() * sizeof(csg::Tape::Instr));
    tapeStagingBuf.Unmap();  

    // Allocate the command buffer.
    VkCommandBuffer cmd;
    auto allocInfo = vkw::init::CommandBufferAllocateInfo(m_cmdPool);
    VK_CHECK(vkAllocateCommandBuffers(m_device->logicalDevice, &allocInfo, &cmd));
    // Begin the command.
    auto beginInfo = vkw::init::CommandBufferBeginInfo(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
    
    // Record the command 
    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = tapeStagingBuf.size;
    vkCmdCopyBuffer(cmd, tapeStagingBuf.buffer, m_voxels->tapeBuffer.buffer, 1, &region);
    
    // End the command
    VK_CHECK(vkEndCommandBuffer(cmd));
    // Submit.
    auto submitInfo = vkw::init::SubmitInfo(&cmd);
    VK_CHECK(vkQueueSubmit(m_queue, 1, &submitInfo, m_fence));

    // Wait for the command to finish.
    VK_CHECK(vkWaitForFences(m_device->logicalDevice, 1, &m_fence, true, 1000000000));
    VK_CHECK(vkResetFences(m_device->logicalDevice, 1, &m_fence));

    tapeStagingBuf.Cleanup();
}

void Voxelizer::Voxelize(VkSemaphore waitSem, float time) 
{
    // No need to actually create the root node,
    // as 0 initialized fields are correct.
    // However we have to encode the number of nodes on level 0.
    m_voxels->interiorNodeCount[0] = 1;
    for (uint32_t i = 0; i < m_voxels->gridLevels; i++) {
        VoxelizeLevel(i, waitSem, time);
    }
}