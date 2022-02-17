#include "vk_wrapper/buffer.h"
#include "vk_wrapper/vk_check.h"


void vkw::Buffer::Init(vkw::Device* dev) 
{
    this->device = dev;
}

void vkw::Buffer::Cleanup() 
{
    assert(buffer != VK_NULL_HANDLE);
    vmaDestroyBuffer(device->vmaAllocator, buffer, allocation);
}

void vkw::Buffer::Allocate(size_t size, VkBufferUsageFlags bufferUsage, VmaMemoryUsage memUsage) 
{
    assert(size > 0);
    assert(size <= (1 << 30)); // more than this freezes my computer
    this->size = size;

    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.pNext = nullptr;
    info.size = size;
    info.usage = bufferUsage;
    
    VmaAllocationCreateInfo vmaInfo = {};
    vmaInfo.usage = memUsage;

    VK_CHECK(vmaCreateBuffer(device->vmaAllocator, &info, &vmaInfo, &buffer, &allocation, nullptr));
}

void* vkw::Buffer::Map() 
{
    void* ptr;
    VK_CHECK(vmaMapMemory(device->vmaAllocator, allocation, &ptr));    
    return ptr;
}

void vkw::Buffer::Unmap() 
{
    vmaUnmapMemory(device->vmaAllocator, allocation);    
}


