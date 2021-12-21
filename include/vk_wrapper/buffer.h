#pragma once
#include "third_party/vk_mem_alloc.h"


namespace vkw
{
    class Buffer
    {
    public:
        VmaAllocator allocator;

        VkBuffer buffer;
        VmaAllocation allocation;
        size_t size;

        void Init(VmaAllocator allocator);
        void Allocate(size_t size, VkBufferUsageFlags bufferUsage, VmaMemoryUsage memUsage);
        void Cleanup();

        void* Map();
        void Unmap();
    };
}