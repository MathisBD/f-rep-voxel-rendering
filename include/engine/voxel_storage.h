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

    uint32_t gridLevels;
    std::vector<uint32_t> gridDims;

    // These are GPU buffers.
    vkw::Buffer nodeBuffer;
    vkw::Buffer childBuffer;
    vkw::Buffer voxelBuffer;

    // The sizes are in bytes.
    uint32_t NodeSize(uint32_t level) {
        return 4 + glm::pow(gridDims[level], 3) / 4;
    }
    uint32_t NodeOfsCLIdx(uint32_t level) {
        return 0;
    }
    uint32_t NodeOfsMask(uint32_t level) {
        return 4;
    }
    uint32_t NodeOfsMaskPC(uint32_t level) {
        return 4 + glm::pow(gridDims[level], 3) / 8;
    }

    uint32_t ChildListSize(uint32_t level) {
        return 4 * glm::pow(gridDims[level], 3);
    }
};  