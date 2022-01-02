#pragma once
#include <vector>
#include <stdint.h>
#include "vk_wrapper/buffer.h"
#include <glm/glm.hpp>


class VoxelStorage
{
public:
    struct Voxel {
        glm::vec3 normal; 
        uint32_t materialIndex;
    };

    // These should be filled before calling Init().
    // gridDims[i] is the number of VOXELS in each direction for nodes
    // at level i (not the number of vertices in each direction).
    std::vector<uint32_t> gridDims;
    glm::vec3 lowVertex;
    float worldSize;
    
    uint32_t gridLevels;
    
    // This is the dimension of the finest grid.
    // Example : for a 3-level grid (3, 4, 2), 
    // the fine grid dimension is 24. 
    uint32_t fineGridDim;

    // The number of nodes in each level.
    // Leaf nodes are counted but not voxels.
    std::vector<uint32_t> nodeCount;

    // These are GPU buffers.
    vkw::Buffer nodeBuffer;
    vkw::Buffer childBuffer;
    vkw::Buffer voxelBuffer;

    void Init(VmaAllocator allocator) 
    {
        gridLevels = gridDims.size();

        fineGridDim = 1;
        for (auto dim : gridDims) {
            fineGridDim *= dim;
        }

        nodeCount = std::vector<uint32_t>(gridLevels, 0);

        nodeBuffer.Init(allocator);
        childBuffer.Init(allocator);
        voxelBuffer.Init(allocator);
    }

    // The coordinates are given in the finest grid dimensions.
    inline glm::vec3 WorldPosition(glm::u32vec3 pos) const 
    {
        return lowVertex + glm::vec3(pos) * worldSize / (float)fineGridDim;    
    }

    // The sizes are in bytes.
    uint32_t NodeSize(uint32_t level) {
        return 16 + glm::pow(gridDims[level], 3) / 4;
    }
    uint32_t NodeOfsCLIdx(uint32_t level) {
        return 0;
    }
    uint32_t NodeOfsCoords(uint32_t level) {
        return 4;
    }
    uint32_t NodeOfsMask(uint32_t level) {
        return 16;
    }
    uint32_t NodeOfsMaskPC(uint32_t level) {
        return 16 + glm::pow(gridDims[level], 3) / 8;
    }

    uint32_t ChildListSize(uint32_t level) {
        return 4 * glm::pow(gridDims[level], 3);
    }
};  