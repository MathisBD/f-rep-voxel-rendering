#pragma once
#include <vulkan/vulkan.h>
#include "engine/renderer.h"
#include "engine/cleanup_queue.h"

class Engine
{
public:
    void Init();
    void Run();
    void Cleanup();
private:
    bool m_isInitialized { false };
   
    VkExtent2D m_windowExtent { 1700, 900 };
    struct SDL_Window* m_window { nullptr };

    // Vulkan core
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkPhysicalDevice m_gpu;
    VkDevice m_device;
    VkSurfaceKHR m_surface;

    // Engine components
    CleanupQueue m_cleanupQueue;
    Renderer m_renderer;

    void InitSDL();
    vkb::Device InitVulkanCore();
    void Draw();
};