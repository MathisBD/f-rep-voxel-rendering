#include "shapes.h"


csg::Expr Shapes::Sphere(const glm::vec3& center, float radius) 
{
    auto x = csg::X();
    auto y = csg::Y();
    auto z = csg::Z();

    auto px = x - center.x;
    auto py = y - center.y;
    auto pz = z - center.z;

    return radius * radius - (px*px + py*py + pz*pz);
}

csg::Expr Shapes::TangleCube(const glm::vec3& center, float scale) 
{
    auto x = csg::X();
    auto y = csg::Y();
    auto z = csg::Z();
    
    auto px = (x - center.x) / scale;
    auto py = (y - center.y) / scale;
    auto pz = (z - center.z) / scale;
    
    auto px2 = px*px;
    auto py2 = py*py;
    auto pz2 = pz*pz;

    return -1 * ((px2*px2 + py2*py2 + pz2*pz2) - 8 * (px2 + py2 + pz2) + 25);
}

/*Shapes::density_t Shapes::BarthSextic(const glm::vec3& center, float scale) {
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
}*/
