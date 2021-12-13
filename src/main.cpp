#include "engine/engine.h"


int main()
{
    Engine engine;
    engine.Init();
    engine.Run();
    engine.Cleanup();
    return 0;
}