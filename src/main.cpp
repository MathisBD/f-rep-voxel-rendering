#include "engine/vk_engine.h"


int main()
{
    VkEngine engine;
    engine.Init();
    engine.Run();
    engine.Cleanup();
    return 0;
}