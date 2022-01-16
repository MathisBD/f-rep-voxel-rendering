#include "vk_wrapper/device.h"

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
