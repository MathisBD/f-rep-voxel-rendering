#include "vk_wrapper/initializers.h"


VkDeviceQueueCreateInfo vkw::init::DeviceQueueCreateInfo(
    uint32_t queueFamily,
    float queuePriority = 0.0f) 
{
    VkDeviceQueueCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    info.pNext = nullptr;

    info.queueFamilyIndex = queueFamily;
    info.queueCount = 1;
    info.pQueuePriorities = &queuePriority;

    return info;    
}