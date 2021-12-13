#pragma once
#include <vulkan/vulkan.h>


class Frame
{
public:
    void Init(
        const VkDevice& device,
        uint32_t graphicsQueueFamily);
    void Cleanup(const VkDevice& device);

    const VkCommandBuffer& GetCommandBuffer() { return m_commandBuffer; };
    const VkFence& GetRenderFence() { return m_renderFence; };
    const VkSemaphore& GetPresentSemaphore() { return m_presentSemaphore; };
    const VkSemaphore& GetRenderSemaphore() { return m_renderSemaphore; };
private:
    VkCommandPool m_commandPool;
    VkCommandBuffer m_commandBuffer;

    VkSemaphore m_renderSemaphore;
    VkSemaphore m_presentSemaphore;
    VkFence m_renderFence;
};