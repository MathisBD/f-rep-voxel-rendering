#pragma once
#include <vulkan/vulkan.h>
#include "engine/swapchain.h"
#include "engine/frame.h"
#include "vk_wrapper/device.h"



class Renderer
{
public:
    VkDevice device;
    Swapchain swapchain;

    void Init(const vkw::Device* dev, VkSurfaceKHR surface, VkExtent2D windowExtent);
    void Cleanup();
    void Draw(VkPipeline pipeline);
private:
    static const size_t FRAME_OVERLAP = 2;

    uint64_t m_frameNumber;
    VkExtent2D m_windowExtent;
    
    VkQueue m_graphicsQueue;
    uint32_t m_graphicsQueueFamily;
    
    Frame m_frames[FRAME_OVERLAP];

    Frame& CurrentFrame();

    void BuildRenderCommand(VkCommandBuffer cmd, uint32_t swapchainImgIdx);
    void SubmitRenderCommand(VkCommandBuffer cmd);
    void PresentImage(uint32_t swapchainImgIdx);
};