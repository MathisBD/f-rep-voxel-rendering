#include "application.h"
#include "shapes.h"


int main()
{
    Application::Params params;
    params.enableValidationLayers = false;
    params.enableShaderDebugPrintf = false;
    params.printFPS = false;
    params.gridDims = { 16, 4, 4, 4 };
    params.shape = Shapes::TangleCube({0, 0, 0}, 4);

    Application app(params);
    app.Init();
    app.Run();
    app.Cleanup();
    return 0;
}