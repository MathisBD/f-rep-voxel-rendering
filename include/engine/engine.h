#pragma once
#include <vulkan/vulkan.h>
#include "engine/swapchain.h"
#include "engine/cleanup_queue.h"
#include "engine/frame.h"


class Engine
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

    // Vulkan core
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkPhysicalDevice m_gpu;
    VkDevice m_device;
    VkSurfaceKHR m_surface;

    VkQueue m_graphicsQueue;
    uint32_t m_graphicsQueueFamily;

    // Engine components
    CleanupQueue m_cleanupQueue;
    Swapchain m_swapchain;
    Frame m_frame;

    // initialization
    void InitVulkanCore();
    void InitSwapchain();

    // Main draw logic
    void Draw();
};