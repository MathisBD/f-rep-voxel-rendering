#pragma once
#include <functional>
#include <glm/glm.hpp>


class Shapes
{
public:
    typedef std::function<float(float, float, float)> density_t;

    static density_t Sphere(const glm::vec3& center, float radius);
    static density_t TangleCube(const glm::vec3& center, float scale);
    static density_t BarthSextic(const glm::vec3& center, float scale);
};