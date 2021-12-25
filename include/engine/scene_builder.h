#pragma once
#include "engine/voxel_storage.h"
#include <functional>
#include "engine/cube_grid.h"


class SceneBuilder
{
public:
    VoxelStorage voxels;
    // This grid has the finest dimension (i.e. the
    // product of the dimensions of every grid level).
    CubeGrid fineGrid;
    std::function<float(float, float, float)> density;

    void Init(VmaAllocator vmaAllocator);
    void CreateVoxels(
        std::function<float(float, float, float)>&& density,
        CubeGrid& fineGrid);
    void Cleanup();
private:
    VmaAllocator m_allocator;
    
    // These are CPU buffers.
    // They are copied into the corresponding GPU buffer.
    vkw::Buffer m_nodeStagingBuffer;
    vkw::Buffer m_childStagingBuffer;
    vkw::Buffer m_voxelStagingBuffer;

    void CopyStagingBuffers();
};