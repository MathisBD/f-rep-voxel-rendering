#pragma once
#include <glm/glm.hpp>
#include <stdint.h>


struct CubeGrid
{
public:
    // number of vertices along each axis (including endpoints)
    uint32_t dim;
    // world position of the grid
    glm::vec3 lowVertex;
    float worldSize;
    float cellSize;

    CubeGrid() {};

    CubeGrid(uint32_t dim, const glm::vec3& low, float size) :
        dim(dim), lowVertex(low), worldSize(size), cellSize(size / (dim-1)) {}
    
    inline glm::vec3 WorldPosition(glm::u32vec3 pos) const 
    {
        assert(pos.x < dim && pos.y < dim && pos.z < dim);
        return lowVertex + glm::vec3(pos) * cellSize;    
    }
};