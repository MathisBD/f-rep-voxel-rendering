#pragma once
#include <vulkan/vulkan.h>
#include "engine/renderer.h"
#include "utils/function_queue.h"
#include "vk_wrapper/device.h"
#include "vk_wrapper/shader.h"
#include "vk_wrapper/descriptor.h"
#include "vk_wrapper/buffer.h"
#include "third_party/vk_mem_alloc.h"
#include <glm/glm.hpp>
#include "engine/input_manager.h"


class EngineBase
{
public:
    virtual void Init();
    void Run();
    virtual void Cleanup();
protected:
    // These should be set by the derived classes.
    bool m_enableValidationLayers = false;
    bool m_enableShaderDebugPrintf = false;

    // Window
    VkExtent2D m_windowExtent { 1700, 900 };
    struct SDL_Window* m_window { nullptr };
    VkSurfaceKHR m_surface;

    // Vulkan core
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    vkw::Device m_device;
    
    // Engine components
    FunctionQueue m_cleanupQueue;
    InputManager m_inputManager;
    vkw::DescriptorLayoutCache m_descCache;
    vkw::DescriptorAllocator m_descAllocator;
    struct {
        VkQueue queue;
        VkCommandPool pool;
        VkFence fence;
    } m_immUploadCtxt;
    

    void InitSDL();
    void InitVulkanCore();
    void InitVma();
    void InitImmUploadCtxt();
    
    // The derived classes should implement this.
    virtual void Draw() = 0;
    
    void ImmediateSubmit(std::function<void(VkCommandBuffer)>&& record);
};