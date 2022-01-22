#include "application.h"
#include "shapes.h"
#include "vk_wrapper/shader.h"
#include "utils/string_utils.h"
#include <sstream>

    
int main()
{
    Application::Params params;
    params.enableValidationLayers = true;
    params.enableShaderDebugPrintf = false;
    params.voxelizeRealTime = false;
    params.printFPS = false;
    params.printHardwareInfo = false;
    params.gridDims = { 4, 4, 4 };
    params.temporalSampleCount = 3;
    
    params.shape = TangleCube();
    
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
}

/*int main()
{
    vkw::ShaderCompiler compiler(nullptr, "/home/mathis/src/f-rep-voxel-rendering/shaders/");
    compiler.Compile("renderer/triangle.vert", vkw::ShaderCompiler::Stage::VERT);    
    return 0;
}*/