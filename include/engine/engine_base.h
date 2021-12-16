#pragma once
#include <vulkan/vulkan.h>
#include "engine/renderer.h"
#include "engine/cleanup_queue.h"
#include "vk_wrapper/device.h"
#include "vk_wrapper/shader.h"

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
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline; 

    void InitSDL();
    void InitVulkanCore();
    void InitPipelines();
    void Draw();
};