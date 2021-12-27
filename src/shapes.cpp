#include "shapes.h"
#include <glm/gtx/norm.hpp>


Shapes::density_t Shapes::Sphere(const glm::vec3& center, float radius) 
{
    return [=] (float x, float y, float z) {
        glm::vec3 pos(x, y, z);
        return radius*radius - glm::distance2(center, pos);
    };
}

Shapes::density_t Shapes::TangleCube(const glm::vec3& center, float scale) 
{
    return [=] (float x, float y, float z) {
        glm::vec3 pos(x, y, z);
        pos = (pos - center) / scale;
        return -(glm::length2(pos * pos) - 8 * glm::length2(pos) + 25);
    };  
}

Shapes::density_t Shapes::BarthSextic(const glm::vec3& center, float scale) {
    float t = (1 + glm::sqrt(5)) / 2;

    return [=] (float x, float y, float z) {
        glm::vec3 pos(x, y, z);
        pos = (pos - center) / scale;
        glm::vec3 p2 = pos * pos;

        float t2 = t*t;
        float res = 4 * (t2*p2.x - p2.y) * (t2*p2.y - p2.z) * (t2*p2.z - p2.x) -
            (1 + 2*t) * (p2.x + p2.y + p2.z - 1) * (p2.x + p2.y + p2.z - 1);
        return res;    
    };
}
