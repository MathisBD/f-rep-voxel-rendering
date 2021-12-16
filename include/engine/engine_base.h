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
    void Init();
    void Run();
    void Cleanup();
private:
    typedef struct {
        glm::vec4 color;
    } GPUCameraData;

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
    vkw::DescriptorLayoutCache m_descriptorCache;
    vkw::DescriptorAllocator m_descriptorAllocator;
    VmaAllocator m_vmaAllocator;

    VkDescriptorSet m_firstSet;
    vkw::Buffer m_cameraBuffer;

    void InitSDL();
    void InitVulkanCore();
    void InitVma();
    void InitPipelines();
    void Draw();
};