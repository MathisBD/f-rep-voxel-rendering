#include "engine/voxelizer.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/shader.h"
#include <cstring>
#include "vk_wrapper/spec_constants.h"
#include <vector>
#include "utils/num_utils.h"


void Voxelizer::Init(
    vkw::Device* device, 
    vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
    VoxelStorage* voxels) 
{
    m_device = device;
    m_descAllocator = descAllocator;
    m_descCache = descCache;
    m_voxels = voxels;
    
    InitCommands();
    InitSynchronization();
    InitBuffers();

    AllocateGPUBuffers();
    ZeroOutNodeBuffer();
    UploadTape();

    InitDescSets();

    for (size_t tapeSize = 128; tapeSize <= 4096; tapeSize *= 2) {
        for (size_t slotCount = 64; slotCount <= 128; slotCount *= 2) {
            m_shaders.push_back(CreateShaderVariant(slotCount, tapeSize));
        }
    }
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
    m_paramsBuffer.Init(m_device);
    m_paramsBuffer.Allocate(
        sizeof(ShaderParams), 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_device->NameObject(m_paramsBuffer.buffer, "voxelizer params buffer");
    m_cleanupQueue.AddFunction([=] { m_paramsBuffer.Cleanup(); });

    // Counters buffer
    m_countersBuffer.Init(m_device);
    m_countersBuffer.Allocate(
        sizeof(ShaderCounters),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_device->NameObject(m_countersBuffer.buffer, "voxelizer counters buffer");
    m_cleanupQueue.AddFunction([=] { m_countersBuffer.Cleanup(); });

    // Stats buffer
    m_statsBuffer.Init(m_device);
    m_statsBuffer.Allocate(
        sizeof(ShaderStats),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_device->NameObject(m_statsBuffer.buffer, "voxelizer stats buffer");
    m_cleanupQueue.AddFunction([=] { m_statsBuffer.Cleanup(); });
}

void Voxelizer::InitDescSets() 
{
    // Descriptor Sets
    m_descSets = std::vector<VkDescriptorSet>(1);
    m_descSetLayouts = std::vector<VkDescriptorSetLayout>(1);

    // Descriptor Set 0.
    auto paramsInfo = vkw::init::DescriptorBufferInfo(
        m_paramsBuffer.buffer, 0, m_paramsBuffer.size);
    auto nodeInfo = vkw::init::DescriptorBufferInfo(
        m_voxels->nodeBuffer.buffer, 0, m_voxels->nodeBuffer.size);
    auto tapeInfo = vkw::init::DescriptorBufferInfo(
        m_voxels->tapeBuffer.buffer, 0, m_voxels->tapeBuffer.size);
    auto countersInfo = vkw::init::DescriptorBufferInfo(
        m_countersBuffer.buffer, 0, m_countersBuffer.size);
    auto statsInfo = vkw::init::DescriptorBufferInfo(
        m_statsBuffer.buffer, 0, m_statsBuffer.size);
    vkw::DescriptorBuilder(m_descCache, m_descAllocator)
        .BindBuffer(0, &paramsInfo,     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(1, &nodeInfo,       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(2, &tapeInfo,       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(3, &countersInfo,   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(4, &statsInfo,      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .Build(&m_descSets[0], &m_descSetLayouts[0]);
    m_device->NameObject(m_descSets[0], "voxelizer descriptor set 0");
}

Voxelizer::ShaderVariant Voxelizer::CreateShaderVariant(uint32_t maxSlotCount, uint32_t maxTapeSize) 
{
    assert(NumUtils::IsPowerOf2(maxSlotCount));
    assert(NumUtils::IsPowerOf2(maxTapeSize));
    ShaderVariant variant;
    variant.maxSlotCount = maxSlotCount;
    variant.maxTapeSize = maxTapeSize;

    // Load the shader
    vkw::ShaderCompiler compiler(m_device, "/home/mathis/src/f-rep-voxel-rendering/shaders/");
    compiler.SetConstant("THREAD_GROUP_SIZE_X", (uint32_t)THREAD_GROUP_SIZE_X);
    compiler.SetConstant("THREAD_GROUP_SIZE_Y", (uint32_t)THREAD_GROUP_SIZE_Y);
    compiler.SetConstant("THREAD_GROUP_SIZE_Z", (uint32_t)THREAD_GROUP_SIZE_Z);
    compiler.SetConstant("LEVEL_COUNT", m_voxels->gridLevels);
    compiler.SetConstant("MAX_SLOT_COUNT", (uint32_t)maxSlotCount);
    compiler.SetConstant("MAX_TAPE_SIZE", (uint32_t)maxTapeSize);
    VkShaderModule shader = compiler.Compile(
        "voxelizer/main.comp", vkw::ShaderCompiler::Stage::COMP);
    
    // Pipeline layout
    auto layoutInfo = vkw::init::PipelineLayoutCreateInfo();
    layoutInfo.setLayoutCount = (uint32_t)m_descSetLayouts.size();
    layoutInfo.pSetLayouts = m_descSetLayouts.data();
    VK_CHECK(vkCreatePipelineLayout(m_device->logicalDevice, &layoutInfo, nullptr, &variant.pipelineLayout));
    m_device->NameObject(variant.pipelineLayout, "voxelizer pipeline layout");

    // Pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = nullptr;
    pipelineInfo.layout = variant.pipelineLayout;
    pipelineInfo.stage = vkw::init::PipelineShaderStageCreateInfo(
        VK_SHADER_STAGE_COMPUTE_BIT, shader);
    
    VK_CHECK(vkCreateComputePipelines(m_device->logicalDevice, VK_NULL_HANDLE, 
        1, &pipelineInfo, nullptr, &variant.pipeline));
    m_device->NameObject(variant.pipeline, "voxelizer pipeline");

    // We can destroy the shader right away.
    vkDestroyShaderModule(m_device->logicalDevice, shader, nullptr);
    m_cleanupQueue.AddFunction([=] {
        vkDestroyPipelineLayout(m_device->logicalDevice, variant.pipelineLayout, nullptr);
        vkDestroyPipeline(m_device->logicalDevice, variant.pipeline, nullptr);
    });
    return variant;
}

Voxelizer::ShaderVariant Voxelizer::FindShaderVariant(uint32_t slotCount, uint32_t tapeSize) 
{
    int best = -1;
    for (size_t i = 0; i < m_shaders.size(); i++) {
        if (slotCount <= m_shaders[i].maxSlotCount && 
            tapeSize <= m_shaders[i].maxTapeSize) 
        {
            if (best < 0 || 
                m_shaders[i].maxTapeSize < m_shaders[best].maxTapeSize ||
                (m_shaders[i].maxTapeSize == m_shaders[best].maxTapeSize && 
                m_shaders[i].maxSlotCount < m_shaders[best].maxSlotCount)) {
                best = i;
            }
        }
    }
    assert(best >= 0);
    return m_shaders[best];
}

void Voxelizer::UpdateShaderParams(uint32_t level, float tapeTime) 
{
    ShaderParams* params = (ShaderParams*)m_paramsBuffer.Map();
    params->level = level;
    params->tapeTime = tapeTime;

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

    memcpy(params->constantPool, m_voxels->tape.constantPool.data(),
        m_voxels->tape.constantPool.size() * sizeof(csg::Tape::Instr));

    m_paramsBuffer.Unmap();

    // Also update the stats buffer.
    if (level == 0) {
        ShaderStats* stats = (ShaderStats*)m_statsBuffer.Map();
        for (uint32_t i = 0; i < MAX_LEVEL_COUNT; i++) {
            stats->tapeSizeMax[i] = 0;
            stats->tapeSizeSum[i] = 0;
        }
        stats->tapeSizeSum[0] = m_voxels->tape.instructions.size();
        stats->tapeSizeMax[0] = m_voxels->tape.instructions.size();
        m_statsBuffer.Unmap();
    }
}

void Voxelizer::UpdateShaderCounters(uint32_t level)
{
    ShaderCounters* counters = (ShaderCounters*)m_countersBuffer.Map();
    counters->childCount = 0;
    // The tape index isn't reset between levels.
    if (level == 0) {
        // We pad the tape size up to a multiple of 32 bytes 
        // to allow coalescing shader memory accesses.
        counters->tapeIndex = NumUtils::RoundUpToMultiple(m_voxels->tape.instructions.size() + 1, 32);
    }
    m_countersBuffer.Unmap();
}


void Voxelizer::RecordCmd(VkCommandBuffer cmd, uint32_t level) 
{
    ShaderStats* stats = (ShaderStats*)m_statsBuffer.Map();
    uint32_t tapeSize = stats->tapeSizeMax[level];
    m_statsBuffer.Unmap();
    auto shader = FindShaderVariant(m_voxels->tape.GetSlotCount(), tapeSize);
    //printf("[+] level=%u  tapeSize=%u  shader.maxSlotCount=%u  shader.maxTapeSize=%u\n",
    //    level, tapeSize, shader.maxSlotCount, shader.maxTapeSize);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        shader.pipelineLayout, 
        0, m_descSets.size(), m_descSets.data(), 
        0, nullptr);

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

void Voxelizer::VoxelizeLevel(uint32_t level, VkSemaphore waitSem, float tapeTime) 
{
    // Update the uniform buffer
    UpdateShaderParams(level, tapeTime);
    UpdateShaderCounters(level);

    m_device->QueueBeginLabel(m_queue, "voxelize level " + std::to_string(level));
    {
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
    }
    m_device->QueueEndLabel(m_queue);

    // Wait for the command to finish.
    VK_CHECK(vkWaitForFences(m_device->logicalDevice, 1, &m_fence, true, 1000000000));
    VK_CHECK(vkResetFences(m_device->logicalDevice, 1, &m_fence));

    ShaderCounters* counters = (ShaderCounters*)m_countersBuffer.Map();
    m_voxels->interiorNodeCount[level + 1] = counters->childCount;
    m_countersBuffer.Unmap();
}

void Voxelizer::AllocateGPUBuffers() 
{
    // Start with 1GB
    uint32_t nodeSize = 1 << 29;
    uint32_t tapeSize = 1 << 29;

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
    nodeStagingBuf.Init(m_device);
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
    tapeStagingBuf.Init(m_device);
    tapeStagingBuf.Allocate(m_voxels->tapeBuffer.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

    uint32_t* tapeContents = (uint32_t*)tapeStagingBuf.Map();
    // Tape size
    uint32_t tapeSize = m_voxels->tape.instructions.size();
    memcpy(tapeContents, &tapeSize, sizeof(uint32_t));
    // Tape contents
    memcpy(tapeContents + 1, m_voxels->tape.instructions.data(),
        tapeSize * sizeof(csg::Tape::Instr));
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

void Voxelizer::Voxelize(VkSemaphore waitSem, float tapeTime) 
{
    m_device->QueueBeginLabel(m_queue, "voxelize");
    // No need to actually create the root node,
    // as 0 initialized fields are correct.
    // However we have to encode the number of nodes on level 0.
    m_voxels->interiorNodeCount[0] = 1;
    for (uint32_t i = 0; i < m_voxels->gridLevels; i++) {
        VoxelizeLevel(i, waitSem, tapeTime);
    }
    m_device->QueueEndLabel(m_queue);
}

void Voxelizer::PrintStats() 
{
    ShaderCounters* counters = (ShaderCounters*)m_countersBuffer.Map();
    ShaderStats* stats = (ShaderStats*)m_statsBuffer.Map();
    
    uint32_t totalNodeCount = 0;
    uint32_t totalNodeSize = 0;
    for (uint32_t i = 0; i < m_voxels->gridLevels; i++) {
        totalNodeCount += m_voxels->interiorNodeCount[i];
        totalNodeSize += m_voxels->interiorNodeCount[i] * m_voxels->NodeSize(i);
    }
    printf("[+] Total interior nodes :\n\tcount=%u\tnode buf bytes=%u\n",
        totalNodeCount, totalNodeSize);

    printf("[+] Total tapes :\n\ttape buf bytes=%u\n",
        4 * counters->tapeIndex);
    
    printf("[+] Interior nodes per level :\n");
    for (uint32_t i = 0; i < m_voxels->gridLevels; i++) {
        uint32_t nodes = m_voxels->interiorNodeCount[i];
        printf("\tlevel %u:\tcount=%6u(%2.1f%%)\tnode buf bytes=%10u(%2.1f%%)\n", 
            i, 
            nodes, 100.0f * nodes / (float)totalNodeCount,
            nodes * m_voxels->NodeSize(i), 100.0f * nodes * m_voxels->NodeSize(i) / (float)totalNodeSize);
    }

    printf("[+] Tape stats per level :\n");
    for (uint32_t i = 0; i < m_voxels->gridLevels; i++) {
        printf("\t%u: avg size=%2.1f  max size=%u\n",
            i, 
            stats->tapeSizeSum[i] / (float)m_voxels->interiorNodeCount[i], 
            stats->tapeSizeMax[i]);
    }

    m_countersBuffer.Unmap();
    m_statsBuffer.Unmap();    
}