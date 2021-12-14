#pragma once
#include <vulkan/vulkan.h>


namespace vkw
{
namespace init
{

VkDeviceQueueCreateInfo DeviceQueueCreateInfo(
    uint32_t queueFamily,
    float queuePriority = 0.0f);

}   // init

}   // vkw