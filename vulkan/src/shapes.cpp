#include "shapes.h"


// A rotating screw.
csg::Expr Screw()
{
    using namespace csg;
    auto shape = Box({-15, -2, -2}, {30, 4, 4});

    auto angle = X();
    shape = RotateX(shape, angle);

    auto scale = 2 * Exp(X() / 40);
    shape = ScaleXYZ(shape, {1, scale, scale});

    shape = RotateZ(shape, T() / 2);
    shape = RotateY(shape, T() / 5);
    return shape;
}

// A scaling cube.
csg::Expr ElasticCube()
{
    using namespace csg;
    auto shape = Box({-10, -10, -10}, {20, 20, 20});
    
    auto scale = 1 / (1 + Pow(Y() / 5, 4));
    shape = ScaleX(shape, scale);
    shape = ScaleX(shape, 1 + 0.2 * Sin(Z() + 10 * T()));

    auto radius = (1 + 0.1 * Sin(T())) * 10;
    shape = Diff(shape, Sphere({0, 0, 0}, radius));    
    
    shape = RotateY(shape, T() / 5);
    return shape;
}

csg::Expr TangleCube() 
{
    using namespace csg;
    auto shape = Norm2(Axes() * Axes()) - 8 * Norm2(Axes()) + 25;
    shape = ScaleXYZ(shape, 4);
    //shape = shape + 15 * Sin(T());
    return shape;
}

csg::Expr Morph()
{
    using namespace csg;
    
    auto shape1 = Norm2(Axes() * Axes()) - 8 * Norm2(Axes()) + 25;
    shape1 = ScaleXYZ(shape1, 4);
    
    auto shape2 = Sphere({0, 0, 0}, 10);
    
    auto blend = (1 + Sin(T())) / 2;
    auto shape = blend * shape1 + (1 - blend) * shape2;
    shape = RotateY(shape, T() / 5);
    return shape;
}

csg::Expr Menger(int level)
{
    using namespace csg;

    if (level <= 0) {
        return Box({0, 0, 0}, {1, 1, 1});
    }

    auto block = ScaleXYZ(Menger(level - 1), {1 / 3.0f, 1 / 3.0f, 1});
    auto menger = Empty();
    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            if (!(x == 1 && y == 1)) {
                auto tb = TranslateXYZ(block, {x / 3.0f, y / 3.0f, 0});
                menger = Union(menger, tb);
            }
        }
    }
    return menger;
}

csg::Expr MengerSponge(int level)
{
    using namespace csg;
    auto shape = Menger(level);
    shape = TranslateXYZ(shape, {-0.5, -0.5, -0.5});
    shape = ScaleXYZ(shape, {30, 30, 10});
    shape = Diff(shape, Sphere({10, 10, 0}, 10));
    return shape;
}

csg::Expr TwistedTower(int level, float angle, float scale)
{
    using namespace csg;
    auto box = Box({-0.5, 0, -0.5}, {1, 1, 1});
    if (level <= 0) {
        return box;
    }

    auto shape = TwistedTower(level-1, angle, scale);
    shape = ScaleXYZ(shape, scale);
    shape = RotateY(shape, angle);
    shape = TranslateY(shape, scale);
    return Union(shape, box);
}

csg::Expr OrganicBalls() 
{
    using namespace csg;
    auto s1 = Sphere({10, 0, 0}, 8);
    s1 = TranslateX(s1, 5*Sin(T()));
    auto s2 = Sphere({-10, 0, 0}, 10);

    auto f = Min(Min(X(), Y()), Sqrt(X() * Y()) - 40);

    return f(s1, s2, Z(), T());
}