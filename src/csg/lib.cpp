#include "csg/lib.h"
#include <glm/gtc/constants.hpp>


namespace csg
{
    Vector3 operator+(Vector3 v1, Vector3 v2)
    {
        return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
    }

    Vector3 operator-(Vector3 v1, Vector3 v2)
    {
        return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
    }

    Vector3 operator*(Vector3 v1, Vector3 v2)
    {
        return { v1.x * v2.x, v1.y * v2.y, v1.z * v2.z };
    }
    
    Vector3 operator/(Vector3 v1, Vector3 v2)
    {
        return { v1.x / v2.x, v1.y / v2.y, v1.z / v2.z };
    }
    

    Vector3 operator-(Vector3 v)
    {
        return { -v.x, -v.y, -v.z };
    }

    Vector3 operator*(Vector3 v, Expr e)
    {
        return { v.x * e, v.y * e, v.z * e };
    }

    Vector3 operator*(Expr e, Vector3 v)
    {
        return { e * v.x, e * v.y, e * v.z };
    }
    
    Vector3 operator/(Vector3 v, Expr e)
    {
        return { v.x / e, v.y / e, v.z / e };
    }

    Expr Empty()
    {
        return Expr(1);
    }

    Vector3 Axes()
    {
        return { X(), Y(), Z() };
    }

    Expr Pi() 
    {
        return Expr(glm::pi<float>());
    }

    Expr Square(Expr x) 
    {
        return x*x;    
    }

    Expr Abs(Expr x)
    {
        return Sqrt(Square(x));
    }

    Expr Pow(Expr x, uint32_t p) 
    {
        // base cases
        if (p == 0) {
            return Constant(1.0f);
        }    
        if (p == 1) {
            return x;
        }

        // induction cases
        if (p % 2 == 0) {
            return Square(Pow(x, p / 2));
        }
        else {
            return Square(Pow(x, p / 2)) * x;
        }
    }

    Expr Dot(Vector3 a, Vector3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    Expr Dist(Vector3 a, Vector3 b) 
    {
        return Norm(b - a);
    }

    Expr Dist2(Vector3 a, Vector3 b) 
    {
        return Norm2(b - a);
    }

    Expr Norm(Vector3 a) 
    {
        return Sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
    }

    Expr Norm2(Vector3 a) 
    {
        return a.x*a.x + a.y*a.y + a.z*a.z;
    }

    Expr Union(Expr a, Expr b) 
    {
        return Min(a, b);    
    }

    Expr Intersect(Expr a, Expr b) 
    {
        return Max(a, b);    
    }

    Expr Diff(Expr a, Expr b) 
    {
        return Intersect(a, Complement(b));    
    }

    Expr Complement(Expr a)
    {
        return -a;
    }

    Expr Sphere(Vector3 center, Expr radius)
    {
        return Dist2(Axes(), center) - Square(radius);
    }

    Expr Box(Vector3 lowVertex, Vector3 size) 
    {
        auto a = Square(X() - (lowVertex.x + size.x / 2)) - Square(size.x / 2);    
        auto b = Square(Y() - (lowVertex.y + size.y / 2)) - Square(size.y / 2);    
        auto c = Square(Z() - (lowVertex.z + size.z / 2)) - Square(size.z / 2);  
        return Intersect(a, Intersect(b, c));
    }

    Expr TranslateX(Expr a, Expr dx) 
    {
        return a(X() - dx, Y(), Z(), T());    
    }

    Expr TranslateY(Expr a, Expr dy) 
    {
        return a(X(), Y() - dy, Z(), T());    
    }

    Expr TranslateZ(Expr a, Expr dz) 
    {
        return a(X(), Y(), Z() - dz, T());    
    }

    Expr TranslateXYZ(Expr a, Vector3 da) 
    {
        return a(X() - da.x, Y() - da.y, Z() - da.z, T());    
    }

    Expr ScaleX(Expr a, Expr scaleX)
    {
        return a(X() / scaleX, Y(), Z(), T());
    }

    Expr ScaleY(Expr a, Expr scaleY)
    {
        return a(X(), Y() / scaleY, Z(), T());
    }

    Expr ScaleZ(Expr a, Expr scaleZ)
    {
        return a(X(), Y(), Z() / scaleZ, T());
    }

    Expr ScaleXYZ(Expr a, Expr scale)
    {
        return a(X() / scale, Y() / scale, Z() / scale, T());
    }

    Expr ScaleXYZ(Expr a, Vector3 scale)
    {
        return a(X() / scale.x, Y() / scale.y, Z() / scale.z, T());
    }
    
    Expr RotateX(Expr a, Expr angleX)
    {
        auto c = Cos(-angleX);
        auto s = Sin(-angleX);
        return a(X(), c * Y() - s * Z(), s * Y() + c * Z(), T());
    }
    
    Expr RotateY(Expr a, Expr angleY)
    {
        auto c = Cos(-angleY);
        auto s = Sin(-angleY);
        return a(c * X() + s * Z(), Y(), -s * X() + c * Z(), T());
    }

    Expr RotateZ(Expr a, Expr angleZ)
    {
        auto c = Cos(-angleZ);
        auto s = Sin(-angleZ);
        return a(c * X() - s * Y(), s * X() + c * Y(), Z(), T());
    }
}

