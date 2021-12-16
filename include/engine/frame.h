#pragma once
#include <vulkan/vulkan.h>
#include "vk_wrapper/device.h"


class Frame
{
public:
    VkDevice device;
    
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkSemaphore renderFinishedSem;
    VkSemaphore imageReadySem;
    VkFence renderFinishedFence;

    void Init(const vkw::Device* dev);
    void Cleanup();
};