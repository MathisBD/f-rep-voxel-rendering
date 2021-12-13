#pragma once
#include <vulkan/vulkan.h>


class Frame
{
public:
    void Init(
        const VkDevice& device,
        uint32_t graphicsQueueFamily);
    void Cleanup(const VkDevice& device);

    const VkCommandBuffer& GetCommandBuffer() const { return m_commandBuffer; };
    const VkFence& GetRenderFinishedFence() const { return m_renderFinishedFence; };
    const VkSemaphore& GetImageReadySem() const { return m_imageReadySem; };
    const VkSemaphore& GetRenderFinishedSem() const { return m_renderFinishedSem; };
private:
    VkCommandPool m_commandPool;
    VkCommandBuffer m_commandBuffer;

    VkSemaphore m_renderFinishedSem;
    VkSemaphore m_imageReadySem;
    VkFence m_renderFinishedFence;
};