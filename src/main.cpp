#include "application.h"
#include "csg/lib.h"


csg::Expr Shape()
{
    using namespace csg;
    auto shape = Box({-20, -2, -2}, {40, 4, 4});

    auto angle = X();
    shape = RotateX(shape, angle);

    auto scale = (X() + 30) / 20;
    shape = ScaleXYZ(shape, {1, scale, scale});
    return shape;
}

int main()
{
    Application::Params params;
    params.enableValidationLayers = false;
    params.enableShaderDebugPrintf = false;
    params.useGPUVoxelizer = true;
    params.printFPS = false;
    params.printHardwareInfo = true;
    params.gridDims = { 16, 16, 4, 4 };
    params.shape = Shape();
    
    csg::Tape tape(params.shape);
    tape.Print();
    
    Application app(params);
    app.Init();
    app.Run();
    app.Cleanup();
    return 0;
}