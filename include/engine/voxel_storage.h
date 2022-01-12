#pragma once
#include <vector>
#include <stdint.h>
#include "vk_wrapper/buffer.h"
#include <glm/glm.hpp>
#include "csg/tape.h"


class VoxelStorage
{
public:
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

    // The number of interior nodes in each level.
    std::vector<uint32_t> interiorNodeCount;

    csg::Tape tape;

    // These are GPU buffers.
    vkw::Buffer nodeBuffer;
    vkw::Buffer tapeBuffer;

    void Init(VmaAllocator allocator, csg::Expr shape) 
    {
        gridLevels = gridDims.size();

        fineGridDim = 1;
        for (auto dim : gridDims) {
            fineGridDim *= dim;
        }

        interiorNodeCount = std::vector<uint32_t>(gridLevels, 0);
        tape = csg::Tape(shape);

        nodeBuffer.Init(allocator);
        tapeBuffer.Init(allocator);
    }

    void Cleanup()
    {
        nodeBuffer.Cleanup();
        tapeBuffer.Cleanup();
    }

    // The coordinates are given in the finest grid dimensions.
    inline glm::vec3 WorldPosition(glm::u32vec3 pos) const 
    {
        return lowVertex + glm::vec3(pos) * worldSize / (float)fineGridDim;    
    }

    // The sizes are in bytes.
    uint32_t NodeSize(uint32_t level) {
        if (level == gridLevels - 1) {
            return 16 + glm::pow(gridDims[level], 3) / 4;
        }
        else {
            return 16 + 17 * (glm::pow(gridDims[level], 3) / 4);
        }
    }
    uint32_t NodeOfsTapeIdx(uint32_t level) {
        return 0;
    }
    uint32_t NodeOfsCoords(uint32_t level) {
        return 4;
    }
    uint32_t NodeOfsLeafMask(uint32_t level) {
        return 16;
    }
    uint32_t NodeOfsInteriorMask(uint32_t level) {
        assert(level < gridLevels - 1);
        return 16 + glm::pow(gridDims[level], 3) / 8;
    }
    uint32_t NodeOfsChildList(uint32_t level) {
        assert(level < gridLevels - 1);
        return 16 + 2 * (glm::pow(gridDims[level], 3) / 8);
    }
};  