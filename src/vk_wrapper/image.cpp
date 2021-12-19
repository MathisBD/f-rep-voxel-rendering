#include "vk_wrapper/image.h"
#include "vk_wrapper/vk_check.h"



void vkw::Image::Init(VmaAllocator allocator) 
{
    this->allocator = allocator;
}

void vkw::Image::Cleanup() 
{
    vmaDestroyImage(allocator, image, allocation);
}

void vkw::Image::Allocate(
    VkExtent2D extent, 
    VkFormat format, 
    VkImageUsageFlags imgUsage, 
    VmaMemoryUsage memUsage,
    VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    const std::vector<uint32_t>* pQueueFamilies = nullptr) 
{
    this->extent = extent;
    this->format = format;

    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.pNext = nullptr;
    
    // we support 2D images for now.
    info.extent.width = extent.width;
    info.extent.height = extent.height;
    info.extent.depth = 1;
    info.format = format;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT; // no multisampling

    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = imgUsage;

    switch (sharingMode) {
    case VK_SHARING_MODE_EXCLUSIVE:
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        break;
    case VK_SHARING_MODE_CONCURRENT:
        info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        assert(pQueueFamilies);
        info.queueFamilyIndexCount = pQueueFamilies->size();
        info.pQueueFamilyIndices = pQueueFamilies->data();
        break;
    default:
        assert(false);
    }

    VmaAllocationCreateInfo vmaInfo = {};
    vmaInfo.usage = memUsage;

    VK_CHECK(vmaCreateImage(allocator, &info, &vmaInfo, &image, &allocation, nullptr));
}


void vkw::Image::ChangeLayout(
    VkCommandBuffer cmd,
    VkImageLayout oldLayout, VkImageLayout newLayout,
    VkPipelineStageFlags srcStages, VkPipelineStageFlags dstStages, 
    VkAccessFlags srcAccess, VkAccessFlags dstAccess) 
{
    VkImageSubresourceRange range = {};
    range.baseArrayLayer = 0;
    range.layerCount = 1;
    range.baseMipLevel = 0;
    range.levelCount = 1;
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext = nullptr;

    barrier.image = image;
    barrier.subresourceRange = range;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;

    vkCmdPipelineBarrier(cmd,
        srcStages, dstStages,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}

void vkw::Image::CopyFromBuffer(
    VkCommandBuffer cmd,
    const vkw::Buffer* buffer) 
{
    VkBufferImageCopy copy = {};
    copy.bufferOffset = 0;
    copy.bufferImageHeight = 0;
    copy.bufferRowLength = 0;

    copy.imageExtent.width = extent.width;
    copy.imageExtent.height = extent.height;
    copy.imageExtent.depth = 1;
    copy.imageOffset = { 0, 0, 0 };

    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.baseArrayLayer = 0;
    copy.imageSubresource.layerCount = 1;
    copy.imageSubresource.mipLevel = 0;

    vkCmdCopyBufferToImage(cmd, 
        buffer->buffer, image, 
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
        1, &copy);
}