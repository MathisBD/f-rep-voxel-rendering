#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "third_party/vk_mem_alloc.h"
#include <string>


namespace vkw
{
    class Device
    {
    public:
        VkPhysicalDevice physicalDevice;
        VkDevice logicalDevice;
        VmaAllocator vmaAllocator;

        VkPhysicalDeviceFeatures features;
        VkPhysicalDeviceProperties properties;

        std::vector<VkQueueFamilyProperties> queueFamilyProperties;
        struct {
            uint32_t graphics;
            uint32_t compute;
            uint32_t transfer;
        } queueFamilies;

        // Extension function pointers
        struct {
            PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT;
        } pfn;

        template <typename T>
        void NameObject(T handle, const std::string& name);

    private:
        void NameObjectHelper(uint64_t handle, const std::string& name, VkObjectType type);
    };
}

