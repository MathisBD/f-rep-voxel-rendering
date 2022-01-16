#pragma once
#include "vk_wrapper/image.h"
#include <stdint.h>
#include "vk_wrapper/device.h"
#include "utils/function_queue.h"


// Essentially a texture, drawed to by the raytracer
// and used by the renderer to create a frame.
class RenderTarget
{
public:
    // This is actually an array of 2D images.
    // There is one layer for each temporal sample.
    vkw::Image image;
    VkImageView view;
    VkSampler sampler;

    uint32_t temporalSampleCount;

    void Init(vkw::Device* device, VkExtent2D windowExtent, uint32_t samples);
    void ZeroOutImage(VkCommandBuffer cmd);
    void Cleanup();
private:
    vkw::Device* m_device;
    VkExtent2D m_windowExtent;
    FunctionQueue m_cleanupQueue;
    
    vkw::Buffer m_stagingBuf;

    void AllocateImage();
    void InitView();
    void InitSampler();
};