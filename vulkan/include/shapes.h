#pragma once
#include "csg/lib.h"


csg::Expr Screw();
csg::Expr ElasticCube();
csg::Expr TangleCube();
csg::Expr Morph();
csg::Expr MengerSponge(int level);
csg::Expr TwistedTower(int level, float angle, float scale);
csg::Expr OrganicBalls();