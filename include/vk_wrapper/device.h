#pragma once
#include <vulkan/vulkan.h>
#include <vector>


namespace vkw
{
    struct Device
    {
        VkPhysicalDevice physicalDevice;
        VkDevice logicalDevice;
        
        VkPhysicalDeviceFeatures features;
        VkPhysicalDeviceProperties properties;

        std::vector<VkQueueFamilyProperties> queueFamilyProperties;
        struct {
            uint32_t graphics;
            uint32_t compute;
            uint32_t transfer;
        } queueFamilies;
    };
}

