#pragma once
#include <vulkan/vulkan.h>
#include "vk_wrapper/swapchain.h"
#include "engine/cleanup_queue.h"
#include "vk_wrapper/descriptor.h"
#include "vk_wrapper/image.h"
#include "engine/render_target.h"


class Renderer
{
public:
    void Init(
        vkw::Device* device, 
        vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
        RenderTarget* target, VkExtent2D windowExtent, VkSurfaceKHR surface);
    void Cleanup();

    // Acquires a swapchain image. 
    // Blocks if no image is available.
    void BeginFrame();
    // Submits the render commands to the graphics queue.
    // Waits on the compute semaphore.
    void Render(VkSemaphore computeSem);
    // Presents the image to the screen
    void EndFrame();

    // Returns the semaphore signaled when the render command ends,
    // to be waited on by the compute command.
    VkSemaphore GetRenderSemaphore() { return m_renderSem; }
private:
    vkw::Device* m_device;
    vkw::DescriptorAllocator* m_descAllocator;
    vkw::DescriptorLayoutCache* m_descCache;
    CleanupQueue m_cleanupQueue;

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

    void InitCommands();
    void InitRenderPass();
    void InitSynchronization();
    void InitPipeline();

    void RecordRenderCmd(VkCommandBuffer cmd);
    void SubmitRenderCmd(VkCommandBuffer cmd, VkSemaphore computeSem);
};