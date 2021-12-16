#include "engine/engine_base.h"


int main()
{
    EngineBase engine;
    engine.Init();
    engine.Run();
    engine.Cleanup();
    return 0;
}