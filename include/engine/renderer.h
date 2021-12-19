#pragma once
#include <vulkan/vulkan.h>
#include "engine/swapchain.h"
#include "engine/frame.h"
#include "vk_wrapper/device.h"
#include <vector>


class Renderer
{
public:

    void Init(const vkw::Device* dev, VkSurfaceKHR surface, VkExtent2D windowExtent);
    void Cleanup();
    void Draw(const DrawInfo* info);
private:

    uint64_t m_frameNumber;
    VkExtent2D m_windowExtent;
    
    Frame m_frames[FRAME_OVERLAP];

    Frame& CurrentFrame();

    void BuildRenderCommand(
        VkCommandBuffer cmd, 
        const DrawInfo* info,
        uint32_t swapchainImgIdx);
    void SubmitRenderCommand(VkCommandBuffer cmd);
    void PresentImage(uint32_t swapchainImgIdx);
};