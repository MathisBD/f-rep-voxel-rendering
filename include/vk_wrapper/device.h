#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <string>


namespace vkw
{
    
    class Device
    {
    public:
        VkPhysicalDevice physicalDevice;
        VkDevice logicalDevice;
        
        VkPhysicalDeviceProperties properties;
        VkPhysicalDeviceFeatures features;
        VkPhysicalDeviceFeatures enabledFeatures;
        std::vector<std::string> supportedExtensions;
        
        std::vector<VkQueueFamilyProperties> queueFamilyProperties;
        struct {
            uint32_t graphics;
            uint32_t compute;
            uint32_t transfer;
        } queueFamilies;

        Device(VkPhysicalDevice physicalDevice);
        ~Device();

        void CreateLogicalDevice(
            VkPhysicalDeviceFeatures requestedFeatures,
            const std::vector<std::string>& requestedExtensions,
            VkQueueFlags requestedQueueTypes);
        bool IsExtensionSupported(const std::string& extension);
    private:
        uint32_t GetQueueFamily(VkQueueFlagBits flags);
    };

} // vkw