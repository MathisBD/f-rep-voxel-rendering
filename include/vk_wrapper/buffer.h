#pragma once
#include "third_party/vk_mem_alloc.h"
#include "vk_wrapper/device.h"


namespace vkw
{
    class Buffer
    {
    public:
        vkw::Device* device;

        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation;
        size_t size;

        void Init(vkw::Device* device);
        void Allocate(size_t size, VkBufferUsageFlags bufferUsage, VmaMemoryUsage memUsage);
        void Cleanup();

        void* Map();
        void Unmap();
    };
}