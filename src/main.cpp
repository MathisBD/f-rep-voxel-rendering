#include "application.h"
#include "shapes.h"
#include "vk_wrapper/shader.h"

    
/*int main()
{
    Application::Params params;
    params.enableValidationLayers = true;
    params.enableShaderDebugPrintf = false;
    params.voxelizeRealTime = false;
    params.printFPS = false;
    params.printHardwareInfo = false;
    params.gridDims = { 16, 8, 4, 4 };
    params.temporalSampleCount = 3;
    
    params.shape = MengerSponge();
    
    csg::Tape tape(params.shape);
    tape.Print();

    uint32_t minMaxOps = 0;
    for (auto i : tape.instructions) {
        if (i.op == csg::Tape::Op::MIN || 
            i.op == csg::Tape::Op::MAX) {
            minMaxOps++;
        }
    }
    printf("[+] Number of min/max instructions : %u\n", minMaxOps);

    Application app(params);
    app.Init();
    app.Run();
    app.Cleanup();
    return 0;
}*/

int main()
{
    vkw::ShaderCompiler compiler(nullptr, 
        "../shaders/renderer/triangle.vert", 
        vkw::ShaderCompiler::Stage::VERT);
    compiler.Compile({});
    return 0;
}