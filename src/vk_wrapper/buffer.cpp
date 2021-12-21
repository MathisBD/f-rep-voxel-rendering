#include "vk_wrapper/buffer.h"
#include "vk_wrapper/vk_check.h"


void vkw::Buffer::Init(VmaAllocator allocator) 
{
    this->allocator = allocator;    
}

void vkw::Buffer::Cleanup() 
{
    vmaDestroyBuffer(allocator, buffer, allocation);
}

void vkw::Buffer::Allocate(size_t size, VkBufferUsageFlags bufferUsage, VmaMemoryUsage memUsage) 
{
    this->size = size;

    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.pNext = nullptr;
    info.size = size;
    info.usage = bufferUsage;
    
    VmaAllocationCreateInfo vmaInfo = {};
    vmaInfo.usage = memUsage;

    VK_CHECK(vmaCreateBuffer(allocator, &info, &vmaInfo, &buffer, &allocation, nullptr));
}

void* vkw::Buffer::Map() 
{
    void* ptr;
    VK_CHECK(vmaMapMemory(allocator, allocation, &ptr));    
    return ptr;
}

void vkw::Buffer::Unmap() 
{
    vmaUnmapMemory(allocator, allocation);    
}


