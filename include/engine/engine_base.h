#pragma once
#include <vulkan/vulkan.h>
#include "engine/renderer.h"
#include "engine/cleanup_queue.h"
#include "vk_wrapper/device.h"
#include "vk_wrapper/shader.h"
#include "vk_wrapper/descriptor.h"
#include "vk_wrapper/buffer.h"
#include "third_party/vk_mem_alloc.h"
#include <glm/glm.hpp>


class EngineBase
{
public:
    virtual void Init();
    void Run();
    virtual void Cleanup();
protected:
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
    vkw::DescriptorLayoutCache m_descriptorCache;
    vkw::DescriptorAllocator m_descriptorAllocator;
    VmaAllocator m_vmaAllocator;
    VkFence m_immFence;
    VkCommandPool m_immCmdPool;

    void InitSDL();
    void InitVulkanCore();
    void InitVma();
    void ImmediateSubmit(std::function<void(VkCommandBuffer)>&& f);

    // The derived classes should implement this.
    virtual void Draw() = 0;
};