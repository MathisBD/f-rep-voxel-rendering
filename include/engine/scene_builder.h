#pragma once
#include "engine/voxel_storage.h"
#include <functional>
#include "engine/cube_grid.h"


class SceneBuilder
{
public:
    VoxelStorage* voxels;
    std::function<float(float, float, float)> density;

    void Init(VmaAllocator vmaAllocator, VoxelStorage* voxels);
    void Cleanup();

    void CreateVoxels(std::function<float(float, float, float)>&& density);
    void AllocateGPUBuffers();
    void CopyStagingBuffers(VkCommandBuffer cmd);
    void CleanupStagingBuffers();

private:
    VmaAllocator m_allocator;
    uint32_t m_actualVoxelCount;

    // These are CPU buffers.
    // They are copied into the corresponding GPU buffer.
    vkw::Buffer m_nodeStagingBuffer;
    vkw::Buffer m_childStagingBuffer;
    vkw::Buffer m_voxelStagingBuffer;
};