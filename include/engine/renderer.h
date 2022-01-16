#pragma once
#include <vulkan/vulkan.h>
#include "vk_wrapper/swapchain.h"
#include "utils/function_queue.h"
#include "vk_wrapper/descriptor.h"
#include "vk_wrapper/image.h"
#include "engine/render_target.h"
#include <glm/glm.hpp>


class Renderer
{
public:
    void Init(
        vkw::Device* device, 
        vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
        RenderTarget* target, VkExtent2D windowExtent, VkSurfaceKHR surface);
    void SignalRenderSem();
    void Cleanup();

    // Acquires a swapchain image. 
    // Blocks if no image is available.
    void BeginFrame();
    // Submits the render commands to the graphics queue.
    // Waits on the given semaphore.
    void Render(VkSemaphore waitSem);
    // Returns the semaphore signaled when the render command ends.
    VkSemaphore GetRenderSem() const { return m_renderSem; }
    // Presents the image to the screen
    void EndFrame();

    void SetClearColor(glm::vec3 color);
private:
    typedef struct {
        uint32_t temporalSampleCount;
        uint32_t _padding_[3];
    } ShaderParams;

    vkw::Device* m_device;
    vkw::DescriptorAllocator* m_descAllocator;
    vkw::DescriptorLayoutCache* m_descCache;
    FunctionQueue m_cleanupQueue;

    RenderTarget* m_target;
    VkExtent2D m_windowExtent;
    VkSurfaceKHR m_surface;

    VkPipeline m_pipeline;
    VkPipelineLayout m_pipelineLayout;
    std::vector<VkDescriptorSet> m_descSets;

    VkQueue m_queue;
    VkCommandPool m_cmdPool;

    // Signaled when the swapchain image is acquired,
    // waited on by the render command.
    VkSemaphore m_imageReadySem;
    // Signaled when the render command is finished,
    // waited on by the compute command.
    VkSemaphore m_renderSem;
    // Signaled when the render command is finished,
    // waited on by the present command.
    VkSemaphore m_presentSem;
    // Signaled when the render command is finished.
    VkFence m_fence;

    VkRenderPass m_renderPass;
    vkw::Swapchain m_swapchain;
    // The index of the swapchain image currently in use.
    uint32_t m_swapCurrImg;
    VkClearValue m_clearValue;

    vkw::Buffer m_shaderParams;

    void InitCommands();
    void InitRenderPass();
    void InitSynchronization();
    void InitBuffers();
    void InitPipeline();

    void UpdateShaderParams();
    void RecordRenderCmd(VkCommandBuffer cmd);
    void SubmitRenderCmd(VkCommandBuffer cmd, VkSemaphore waitSem);
};