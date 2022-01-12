#pragma once
#include <glm/glm.hpp>
#include "csg/expression.h"


class Shapes
{
public:
    static csg::Expr Sphere(const glm::vec3& center, float radius);
    static csg::Expr TangleCube(const glm::vec3& center, float scale);
    static csg::Expr BarthSextic(const glm::vec3& center, float scale);
};