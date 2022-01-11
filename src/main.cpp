#include "application.h"
#include "shapes.h"


int main()
{
    Application::Params params;
    params.enableValidationLayers = false;
    params.enableShaderDebugPrintf = false;
    params.useGPUVoxelizer = true;
    params.printFPS = true;
    params.gridDims = { 16, 8, 8, 4 };
    params.shape = Shapes::TangleCube({0, 0, 0}, 4);

    Application app(params);
    app.Init();
    app.Run();
    app.Cleanup();
    return 0;
}