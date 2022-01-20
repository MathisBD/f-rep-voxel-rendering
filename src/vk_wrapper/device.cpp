#include "vk_wrapper/device.h"
#include <assert.h>


void vkw::Device::NameObjectHelper(uint64_t handle, const std::string& name, VkObjectType type)
{
    if (pfn.vkSetDebugUtilsObjectNameEXT) {
        VkDebugUtilsObjectNameInfoEXT info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        info.pNext = nullptr;

        info.objectHandle = handle;
        info.objectType = type;
        info.pObjectName = name.c_str();
        pfn.vkSetDebugUtilsObjectNameEXT(logicalDevice, &info);
    }
}

template <>
void vkw::Device::NameObject(VkPipeline pipeline, const std::string& name) {
    NameObjectHelper((uint64_t)pipeline, name, VK_OBJECT_TYPE_PIPELINE);
}
template <>
void vkw::Device::NameObject(VkBuffer buffer, const std::string& name) {
    NameObjectHelper((uint64_t)buffer, name, VK_OBJECT_TYPE_BUFFER);
}
template <>
void vkw::Device::NameObject(VkDescriptorSet descSet, const std::string& name) {
    NameObjectHelper((uint64_t)descSet, name, VK_OBJECT_TYPE_DESCRIPTOR_SET);
}
template <>
void vkw::Device::NameObject(VkImage image, const std::string& name) {
    NameObjectHelper((uint64_t)image, name, VK_OBJECT_TYPE_IMAGE);
}
template <>
void vkw::Device::NameObject(VkImageView view, const std::string& name) {
    NameObjectHelper((uint64_t)view, name, VK_OBJECT_TYPE_IMAGE_VIEW);
}
template <>
void vkw::Device::NameObject(VkSampler sampler, const std::string& name) {
    NameObjectHelper((uint64_t)sampler, name, VK_OBJECT_TYPE_SAMPLER);
}

void vkw::Device::CmdBeginLabel(VkCommandBuffer cmd, const std::string& name) 
{
    if (pfn.vkCmdBeginDebugUtilsLabelEXT) {
        VkDebugUtilsLabelEXT info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        info.pNext = nullptr;

        info.pLabelName = name.c_str();
        info.color[0] = info.color[1] = info.color[2] = info.color[3] = 1.0f;
        pfn.vkCmdBeginDebugUtilsLabelEXT(cmd, &info);
    }    
}

void vkw::Device::CmdEndLabel(VkCommandBuffer cmd) 
{
    if (pfn.vkCmdEndDebugUtilsLabelEXT) {
        pfn.vkCmdEndDebugUtilsLabelEXT(cmd);
    }    
}

void vkw::Device::CmdInsertLabel(VkCommandBuffer cmd, const std::string& name) 
{
    if (pfn.vkCmdInsertDebugUtilsLabelEXT) {
        VkDebugUtilsLabelEXT info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        info.pNext = nullptr;
        
        info.pLabelName = name.c_str();
        info.color[0] = info.color[1] = info.color[2] = info.color[3] = 1.0f;
        pfn.vkCmdInsertDebugUtilsLabelEXT(cmd, &info);
    }    
}

void vkw::Device::QueueBeginLabel(VkQueue queue, const std::string& name) 
{
    if (pfn.vkQueueBeginDebugUtilsLabelEXT) {
        VkDebugUtilsLabelEXT info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        info.pNext = nullptr;

        info.pLabelName = name.c_str();
        info.color[0] = info.color[1] = info.color[2] = info.color[3] = 0.3f;
        pfn.vkQueueBeginDebugUtilsLabelEXT(queue, &info);
    }    
}

void vkw::Device::QueueEndLabel(VkQueue queue) 
{
    if (pfn.vkQueueEndDebugUtilsLabelEXT) {
        pfn.vkQueueEndDebugUtilsLabelEXT(queue);
    }    
}

void vkw::Device::QueueInsertLabel(VkQueue queue, const std::string& name) 
{
    if (pfn.vkQueueInsertDebugUtilsLabelEXT) {
        VkDebugUtilsLabelEXT info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        info.pNext = nullptr;
        
        info.pLabelName = name.c_str();
        info.color[0] = info.color[1] = info.color[2] = info.color[3] = 1.0f;
        pfn.vkQueueInsertDebugUtilsLabelEXT(queue, &info);
    }    
}