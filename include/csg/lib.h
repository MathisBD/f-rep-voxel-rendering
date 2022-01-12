#pragma once

#include "csg/expression.h"


namespace csg
{
    class Vector3
    {
    public:
        Expr x;
        Expr y;
        Expr z;

        Vector3(Expr x_, Expr y_, Expr z_) : x(x_), y(y_), z(z_) {}
    };
    
    Vector3 operator+(Vector3 a, Vector3 b);
    Vector3 operator-(Vector3 a, Vector3 b);
    Vector3 operator-(Vector3 a);
    Vector3 operator*(Vector3 a, float constant);
    Vector3 operator*(float constant, Vector3 b);
    Vector3 operator/(Vector3 a, float constant);
    

    // returns a vector whose components 
    // are csg::X(), csg::Y() and csg::Z()
    Vector3 Axis();

    Expr Square(Expr x);
    Expr Abs(Expr x);
    Expr Pow(Expr x, uint32_t p);
    
    Expr Dot(Vector3 a, Vector3 b);
    Expr Norm(Vector3 a);
    Expr Norm2(Vector3 a);
    Expr Dist(Vector3 a, Vector3 b);
    Expr Dist2(Vector3 a, Vector3 b);

    Expr Union(Expr a, Expr b);
    Expr Intersect(Expr a, Expr b);
    Expr Complement(Expr a, Expr b);
    Expr Diff(Expr a, Expr b);

    Expr Sphere(Vector3 center, Expr radius);
    Expr Box(Vector3 lowVertex, Vector3 size);
    
    Expr TranslateX(Expr a, Expr dx);
    Expr TranslateY(Expr a, Expr dy);
    Expr TranslateZ(Expr a, Expr dz);
    Expr TranslateXYZ(Expr a, Vector3 da);
    
    Expr ScaleX(Expr a, Expr scaleX);
    Expr ScaleY(Expr a, Expr scaleY);
    Expr ScaleZ(Expr a, Expr scaleZ);
    Expr ScaleXYZ(Expr a, Vector3 scale);
    
    Expr RotateX(Expr a, Expr angleX);
    Expr RotateY(Expr a, Expr angleY);
    Expr RotateZ(Expr a, Expr angleZ);
}
