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
    

}