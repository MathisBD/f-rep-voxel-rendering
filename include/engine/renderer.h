#pragma once
#include <vulkan/vulkan.h>
#include "engine/swapchain.h"
#include "engine/frame.h"
#include "VkBootstrap.h"


class Renderer
{
public:
    void Init(
        const vkb::Device& vkbDevice,
        const VkExtent2D& windowExtent);
    void Cleanup(const VkDevice& device);

    void Draw(const VkDevice& device);
private:
    static const size_t FRAME_OVERLAP = 2;

    uint64_t m_frameNumber;
    VkExtent2D m_windowExtent;
    
    VkQueue m_graphicsQueue;
    uint32_t m_graphicsQueueFamily;
    
    Swapchain m_swapchain;
    Frame m_frames[FRAME_OVERLAP];

    Frame& CurrentFrame();

    void BuildRenderCommand(const VkCommandBuffer& cmd, uint32_t swapchainImgIdx);
    void SubmitRenderCommand(const VkCommandBuffer& cmd);
    void PresentImage(uint32_t swapchainImgIdx);
};