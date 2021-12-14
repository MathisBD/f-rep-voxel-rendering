#include "vk_wrapper/device.h"
#include "vk_wrapper/initializers.h"
#include <assert.h>
#include "engine/vk_check.h"


vkw::Device::Device(VkPhysicalDevice physDev) 
{
    assert(physDev);
    physicalDevice = physDev;

    // device properties/features
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    vkGetPhysicalDeviceFeatures(physicalDevice, &features);
    
    // queue families
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    assert(queueFamilyCount > 0);
    queueFamilyProperties.resize(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

    // supported extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extCount);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, extensions.data()));
    for (auto ext : extensions) {
        supportedExtensions.push_back(ext.extensionName);
    }
}

vkw::Device::~Device() 
{
    if (logicalDevice) {
        vkDestroyDevice(logicalDevice, nullptr);
    }    
}

void vkw::Device::CreateLogicalDevice(
        VkPhysicalDeviceFeatures requestedFeatures,
        const std::vector<std::string>& requestedExtensions,
        VkQueueFlags requestedQueueTypes) 
{
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
 
    // graphics queue
    queueFamilies.graphics = 0;
    if (requestedQueueTypes & VK_QUEUE_GRAPHICS_BIT) {
        queueFamilies.graphics = GetQueueFamily(VK_QUEUE_GRAPHICS_BIT);
        queueInfos.push_back(vkw::init::DeviceQueueCreateInfo(
            queueFamilies.graphics));
    }
    // compute queue
    queueFamilies.compute = queueFamilies.graphics;
    if (requestedQueueTypes & VK_QUEUE_COMPUTE_BIT) {
        queueFamilies.compute = GetQueueFamily(VK_QUEUE_COMPUTE_BIT);
        if (queueFamilies.compute != queueFamilies.graphics) {
            queueInfos.push_back(vkw::init::DeviceQueueCreateInfo(
                queueFamilies.compute));
        }
    }
    // transfer queue
    queueFamilies.transfer = queueFamilies.graphics;
    if (requestedQueueTypes & VK_QUEUE_TRANSFER_BIT) {
        queueFamilies.transfer = GetQueueFamily(VK_QUEUE_TRANSFER_BIT);
        if (queueFamilies.transfer != queueFamilies.graphics && 
            queueFamilies.transfer != queueFamilies.compute) {
            queueInfos.push_back(vkw::init::DeviceQueueCreateInfo(
                queueFamilies.transfer));
        }
    }
    
}
