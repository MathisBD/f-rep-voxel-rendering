#include "engine/scene_builder.h"



void SceneBuilder::Init(VmaAllocator vmaAllocator) 
{
    m_allocator = vmaAllocator;
}

void SceneBuilder::Cleanup() 
{
    voxels.nodeBuffer.Cleanup();
    voxels.childBuffer.Cleanup();
    voxels.voxelBuffer.Cleanup();
}

void SceneBuilder::CreateVoxels(std::function<float(float, float, float)>&& density) 
{
    this->density = density;
    
    // Create the staging buffers
    m_nodeStagingBuffer.Init(m_allocator);
    m_childStagingBuffer.Init(m_allocator);
    m_voxelStagingBuffer.Init(m_allocator);

    uint32_t voxelCount = glm::pow(voxels->gridDims[0], 3);
    m_nodeStagingBuffer.Allocate(voxels->NodeSize(0), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_childStagingBuffer.Allocate(voxels->ChildListSize(0), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_voxelStagingBuffer.Allocate(voxelCount * sizeof(VoxelStorage::Voxel), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    
    void* node = m_nodeStagingBuffer.Map();
    memset(node, 0, voxels->NodeSize(0));
    uint32_t* mask = (uint32_t*)((size_t)node + voxels->NodeOfsMask(0));
    uint32_t* maskPC = (uint32_t*)((size_t)node + voxels->NodeOfsMaskPC(0));
    uint32_t* childList = (uint32_t*)m_childStagingBuffer.Map();
    VoxelStorage::Voxel* data = (VoxelStorage::Voxel*)m_voxelStagingBuffer.Map(); 

    // Generate the voxel data and mask.
    for (uint32_t x = 0; x < voxels->gridDims[0]; x++) {
        for (uint32_t y = 0; y < voxels->gridDims[0]; y++) {
            for (uint32_t z = 0; z < voxels->gridDims[0]; z++) {
                uint32_t index = z + y * voxels->gridDims[0] + x * voxels->gridDims[0] * voxels->gridDims[0];
                //uint32_t index = MortonEncode({x, y, z});
                
                glm::vec3 pos = voxels->WorldPosition({ x, y, z });
                float d = density(pos.x, pos.y, pos.z);

                if (d >= 0.0f) {
                    uint32_t q = index >> 5;
                    uint32_t r = index & ((1 << 5) - 1);
                    mask[q] |= (1 << r);
                    
                    float eps = 0.001f;
                    data[index].normal = { 
                        (density(pos.x + eps, pos.y, pos.z) - d) / eps,
                        (density(pos.x, pos.y + eps, pos.z) - d) / eps,
                        (density(pos.x, pos.y, pos.z + eps) - d) / eps };
                    data[index].normal = -glm::normalize(data[index].normal);
                    data[index].materialIndex = 0;
                }
            }
        }
    }
    // Compactify the voxel data buffer and the child list
    uint32_t vPos = 0;
    for (uint32_t index = 0; index < voxelCount; index++) {
        uint32_t q = index >> 5;
        uint32_t r = index & ((1 << 5) - 1);
        if (mask[q] & (1 << r)) {
            data[vPos] = data[index];
            childList[vPos] = vPos;
            vPos++;
        }
    }
    m_actualVoxelCount = vPos;

    // Compute the mask partial counts
    uint32_t partialCount = 0;
    for (uint32_t q = 0; q < (voxelCount / 32); q++) {
        partialCount += __builtin_popcount(mask[q]);
        maskPC[q] = partialCount;
    }
    m_nodeStagingBuffer.Unmap();
    m_childStagingBuffer.Unmap();
    m_voxelStagingBuffer.Unmap();
}

void SceneBuilder::AllocateGPUBuffers() 
{
    VkBufferUsageFlags bufferUsage = 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    voxels->nodeBuffer.Allocate(voxels->NodeSize(0), bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    voxels->childBuffer.Allocate(m_actualVoxelCount * sizeof(uint32_t), bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    voxels->voxelBuffer.Allocate(m_actualVoxelCount * sizeof(VoxelStorage::Voxel), bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
}

void SceneBuilder::CopyStagingBuffers(VkCommandBuffer cmd) 
{
    VkBufferCopy nodeRegion;
    nodeRegion.srcOffset = 0;
    nodeRegion.dstOffset = 0;
    nodeRegion.size = voxels->NodeSize(0);
    vkCmdCopyBuffer(cmd, m_nodeStagingBuffer.buffer, voxels->nodeBuffer.buffer, 1, &nodeRegion);

    VkBufferCopy clRegion;
    clRegion.srcOffset = 0;
    clRegion.dstOffset = 0;
    clRegion.size = m_actualVoxelCount * sizeof(uint32_t);
    vkCmdCopyBuffer(cmd, m_childStagingBuffer.buffer, voxels->childBuffer.buffer, 1, &clRegion);

    VkBufferCopy voxelRegion;
    voxelRegion.srcOffset = 0;
    voxelRegion.dstOffset = 0;
    voxelRegion.size = m_actualVoxelCount * sizeof(VoxelStorage::Voxel);
    vkCmdCopyBuffer(cmd, m_voxelStagingBuffer.buffer, voxels->voxelBuffer.buffer, 1, &voxelRegion); 
}

void SceneBuilder::CleanupStagingBuffers() 
{
    m_nodeStagingBuffer.Cleanup();
    m_childStagingBuffer.Cleanup();
    m_voxelStagingBuffer.Cleanup();    
}