#include "csg/lib.h"



csg::Vector3 operator+(csg::Vector3 a, csg::Vector3 b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

csg::Vector3 operator-(csg::Vector3 a, csg::Vector3 b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

csg::Vector3 operator-(csg::Vector3 a)
{
    return { -a.x, -a.y, -a.z };
}

csg::Vector3 operator*(csg::Vector3 a, float constant)
{
    return { a.x * constant, a.y * constant, a.z * constant };
}

csg::Vector3 operator*(float constant, csg::Vector3 b)
{
    return { constant * b.x, constant * b.y, constant * b.z };
}

csg::Vector3 operator/(csg::Vector3 a, float constant)
{
    return { a.x / constant, a.y / constant, a.z / constant };
}

csg::Vector3 csg::Axis()
{
    return { csg::X(), csg::Y(), csg::Z() };
}

csg::Expr csg::Square(csg::Expr x) 
{
    return x*x;    
}

csg::Expr csg::Abs(csg::Expr x)
{
    return csg::Sqrt(csg::Square(x));
}

csg::Expr csg::Pow(csg::Expr x, uint32_t p) 
{
    // base cases
    if (p == 0) {
        return csg::Constant(1.0f);
    }    
    if (p == 1) {
        return x;
    }

    // induction cases
    if (p % 2 == 0) {
        return csg::Square(csg::Pow(x, p / 2));
    }
    else {
        return csg::Square(csg::Pow(x, p / 2)) * x;
    }
}

csg::Expr csg::Dist(csg::Vector3 a, csg::Vector3 b) 
{
    return csg::Norm(b - a);
}

csg::Expr csg::Dist2(csg::Vector3 a, csg::Vector3 b) 
{
    return csg::Norm2(b - a);
}

csg::Expr csg::Norm(csg::Vector3 a) 
{
    return csg::Sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

csg::Expr csg::Norm2(csg::Vector3 a) 
{
    return a.x*a.x + a.y*a.y + a.z*a.z;
}

csg::Expr csg::Union(csg::Expr a, csg::Expr b) 
{
    return csg::Min(a, b);    
}

csg::Expr csg::Intersect(csg::Expr a, csg::Expr b) 
{
    return csg::Max(a, b);    
}

csg::Expr csg::Diff(csg::Expr a, csg::Expr b) 
{
    return csg::Intersect(a, csg::Complement(b));    
}

csg::Expr csg::Complement(csg::Expr a)
{
    return -a;
}

csg::Expr csg::Sphere(csg::Vector3 center, csg::Expr radius)
{
    return csg::Dist2(csg::Axis(), center) - csg::Square(radius);
}

csg::Expr csg::Box(csg::Vector3 lowVertex, csg::Vector3 size) 
{
    auto a = csg::Square(csg::X() - (lowVertex.x + size.x) / 2) - csg::Square(size.x / 2);    
    auto b = csg::Square(csg::Y() - (lowVertex.y + size.y) / 2) - csg::Square(size.y / 2);    
    auto c = csg::Square(csg::Z() - (lowVertex.z + size.z) / 2) - csg::Square(size.z / 2);    
    return csg::Intersect(a, csg::Intersect(b, c));
}

csg::Expr csg::TranslateX(csg::Expr a, csg::Expr dx) 
{
    return a(csg::X() - dx, csg::Y(), csg::Z());    
}

csg::Expr csg::TranslateY(csg::Expr a, csg::Expr dy) 
{
    return a(csg::X(), csg::Y() - dy, csg::Z());    
}

csg::Expr csg::TranslateZ(csg::Expr a, csg::Expr dz) 
{
    return a(csg::X(), csg::Y(), csg::Z() - dz);    
}

csg::Expr csg::TranslateXYZ(csg::Expr a, csg::Vector3 da) 
{
    return a(csg::X() - da.x, csg::Y() - da.y, csg::Z() - da.z);    
}

