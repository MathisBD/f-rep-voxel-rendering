#pragma once
#include "engine/voxel_storage.h"
#include <functional>


class SceneBuilder
{
public:
    void Init(VmaAllocator vmaAllocator, VoxelStorage* voxels);
    void Cleanup();

    void CreateVoxels(std::function<float(float, float, float)>&& density);
    void AllocateGPUBuffers();
    void CopyStagingBuffers(VkCommandBuffer cmd);
private:
    VmaAllocator m_allocator;
    VoxelStorage* m_voxels;
    std::function<float(float, float, float)> m_density;

    uint32_t m_actualVoxelCount = 0;

    // These are CPU buffers.
    // They are copied into the corresponding GPU buffer.
    vkw::Buffer m_nodeStagingBuffer;
    vkw::Buffer m_childStagingBuffer;
    vkw::Buffer m_voxelStagingBuffer;
};