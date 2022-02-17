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
            PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT;
            PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT;
            PFN_vkCmdInsertDebugUtilsLabelEXT vkCmdInsertDebugUtilsLabelEXT;
            PFN_vkQueueBeginDebugUtilsLabelEXT vkQueueBeginDebugUtilsLabelEXT;
            PFN_vkQueueEndDebugUtilsLabelEXT vkQueueEndDebugUtilsLabelEXT;
            PFN_vkQueueInsertDebugUtilsLabelEXT vkQueueInsertDebugUtilsLabelEXT;
        } pfn;

        template <typename T>
        void NameObject(T handle, const std::string& name);

        void CmdBeginLabel(VkCommandBuffer cmd, const std::string& name);
        void CmdEndLabel(VkCommandBuffer cmd);
        void CmdInsertLabel(VkCommandBuffer cmd, const std::string& name);

        void QueueBeginLabel(VkQueue queue, const std::string& name);
        void QueueEndLabel(VkQueue queue);
        void QueueInsertLabel(VkQueue queue, const std::string& name);
    private:
        void NameObjectHelper(uint64_t handle, const std::string& name, VkObjectType type);
    };
}

