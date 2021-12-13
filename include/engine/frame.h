#pragma once
#include <vulkan/vulkan.h>


class Frame
{
public:
    void Init(
        const VkDevice& device,
        uint32_t graphicsQueueFamily);
    void Cleanup(const VkDevice& device);

    const VkCommandBuffer& GetCommandBuffer();
private:
    VkCommandPool m_commandPool;
    VkCommandBuffer m_commandBuffer;
};