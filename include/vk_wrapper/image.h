#pragma once
#include <vulkan/vulkan.h>
#include "third_party/vk_mem_alloc.h"
#include "vk_wrapper/buffer.h"
#include <vector>


namespace vkw
{
    class Image
    {
    public:
        VmaAllocator allocator;

        VkImage image;
        VmaAllocation allocation;
        VkExtent2D extent;
        VkFormat format;

        void Init(VmaAllocator allocator);
        void Allocate(
            VkExtent2D extent, 
            VkFormat foramt,
            VkImageUsageFlags imgUsage, 
            VmaMemoryUsage memUsage,
            VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            const std::vector<uint32_t>* pQueueFamilies = nullptr);
        void Cleanup();

        void ChangeLayout(
            VkCommandBuffer cmd,
            VkImageLayout oldLayout, VkImageLayout newLayout,
            VkPipelineStageFlags srcStages, VkPipelineStageFlags dstStages,
            VkAccessFlags srcAccess, VkAccessFlags dstAccess);
        // The image must be in layout DST_OPTIMAL.
        void CopyFromBuffer(
            VkCommandBuffer cmd,
            const vkw::Buffer* buffer);
    };
}