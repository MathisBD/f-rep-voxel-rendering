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

void SceneBuilder::CreateVoxels(
    std::function<float(float, float, float)>&& density,
    CubeGrid& fineGrid) 
{
    this->fineGrid = fineGrid;
    this->density = density;
    
    // create the voxel storage.
    voxels.gridLevels = 1;
    voxels.gridDims = { fineGrid.dim };
    voxels.nodeBuffer.Init(m_allocator);
    voxels.childBuffer.Init(m_allocator);
    voxels.voxelBuffer.Init(m_allocator);

    // Create the staging buffers
    m_nodeStagingBuffer.Init(m_allocator);
    m_childStagingBuffer.Init(m_allocator);
    m_voxelStagingBuffer.Init(m_allocator);

    uint32_t voxelCount = glm::pow(voxels.gridDims[0], 3);
    m_nodeStagingBuffer.Allocate(voxels.NodeSize(0), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_childStagingBuffer.Allocate(voxels.ChildListSize(0), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_voxelStagingBuffer.Allocate(voxelCount * sizeof(VoxelStorage::Voxel), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    
    void* node = m_nodeStagingBuffer.Map();
    memset(node, 0, voxels.NodeSize(0));
    uint32_t* mask = (uint32_t*)((size_t)node + voxels.NodeOfsMask(0));
    uint32_t* maskPC = (uint32_t*)((size_t)node + voxels.NodeOfsMaskPC(0));
    uint32_t* childList = (uint32_t*)m_childStagingBuffer.Map();
    VoxelStorage::Voxel* data = (VoxelStorage::Voxel*)m_voxelStagingBuffer.Map(); 

    // Generate the voxel data and mask.
    for (uint32_t x = 0; x < voxels.gridDims[0]; x++) {
        for (uint32_t y = 0; y < voxels.gridDims[0]; y++) {
            for (uint32_t z = 0; z < voxels.gridDims[0]; z++) {
                uint32_t index = z + y * voxels.gridDims[0] + x * voxels.gridDims[0] * voxels.gridDims[0];
                //uint32_t index = MortonEncode({x, y, z});
                
                glm::vec3 pos = fineGrid.WorldPosition({ x, y, z });
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
    // Compactify the voxel data buffer
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
    // Compute the mask partial counts
    uint32_t partialCount = 0;
    for (uint32_t q = 0; q < (voxelCount / 32); q++) {
        partialCount += __builtin_popcount(mask[q]);
        maskPC[q] = partialCount;
    }
    stagingData.Unmap();
    stagingMask.Unmap();
    stagingMaskPC.Unmap();
}