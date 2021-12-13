#pragma once
#include <vulkan/vulkan.h>



class VkEngine
{
public:
    void Init();
    void Run();
    void Cleanup();
private:
    bool m_isInitialized { false };
    bool m_frameNumber { 0 };

    VkExtent2D m_windowExtent { 1700, 900 };
    struct SDL_Window* m_window { nullptr };

    void Draw();
};