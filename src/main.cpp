#include "application.h"
#include "shapes.h"


int main()
{
    Application::Params params;
    params.enableValidationLayers = false;
    params.enableShaderDebugPrintf = false;
    params.printFPS = true;
    params.gridDims = { 64 };
    params.shape = Shapes::Sphere({0, 0, 0}, 20);

    Application app(params);
    app.Init();
    app.Run();
    app.Cleanup();
    return 0;
}