#include "engine/engine.h"


int main()
{
    EngineBase engine;
    engine.Init();
    engine.Run();
    engine.Cleanup();
    return 0;
}