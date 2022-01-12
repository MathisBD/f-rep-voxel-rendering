#include "application.h"
#include "shapes.h"


int main()
{
    Application::Params params;
    params.enableValidationLayers = false;
    params.enableShaderDebugPrintf = false;
    params.useGPUVoxelizer = true;
    params.printFPS = false;
    params.printHardwareInfo = true;
    params.gridDims = { 32, 4, 4 };
    params.shape = csg::Max(
        Shapes::Sphere({0, 0, 0}, 20),
        -Shapes::Sphere({10, 10, 10}, 10));

    Application app(params);
    app.Init();
    app.Run();
    app.Cleanup();
    return 0;
}