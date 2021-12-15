#pragma once
#include <vulkan/vulkan.h>
#include "engine/renderer.h"
#include "engine/cleanup_queue.h"
#include <vk_wrapper/device.h>


class EngineBase
{
public:
    void Init();
    void Run();
    void Cleanup();
private:
    // Window
    VkExtent2D m_windowExtent { 1700, 900 };
    struct SDL_Window* m_window { nullptr };
    VkSurfaceKHR m_surface;


    // Vulkan core
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    vkw::Device m_device;
    
    // Engine components
    CleanupQueue m_cleanupQueue;
    Renderer m_renderer;

    void InitSDL();
    void InitVulkanCore();
    void Draw();
};