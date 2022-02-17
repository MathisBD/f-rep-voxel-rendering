#include "vk_wrapper/descriptor.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/initializers.h"


void vkw::DescriptorAllocator::Init(VkDevice device) 
{
    this->device = device;
}

void vkw::DescriptorAllocator::Cleanup() 
{
    for (auto p : m_freePools) {
        vkDestroyDescriptorPool(device, p, nullptr);
    }    
    for (auto p : m_usedPools) {
        vkDestroyDescriptorPool(device, p, nullptr);
    }    
}

void vkw::DescriptorAllocator::ResetPools() 
{
    for (auto p : m_usedPools) {
        vkResetDescriptorPool(device, p, 0);
        m_freePools.push_back(p);
    }    
    m_usedPools.clear();
    m_currentPool = VK_NULL_HANDLE;
}

VkResult vkw::DescriptorAllocator::Allocate(VkDescriptorSet* pSet, VkDescriptorSetLayout layout) 
{
    if (m_currentPool == VK_NULL_HANDLE) {
        m_currentPool = GetFreePool();
        m_usedPools.push_back(m_currentPool);
    }

    VkDescriptorSetAllocateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    info.pNext = nullptr;

    info.descriptorPool = m_currentPool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;

    VkResult res = vkAllocateDescriptorSets(device, &info, pSet);
    switch (res) {
    case VK_ERROR_OUT_OF_POOL_MEMORY:
    case VK_ERROR_FRAGMENTED_POOL:
        // try to allocate in a new pool
        m_currentPool = GetFreePool();
        m_usedPools.push_back(m_currentPool);
        info.descriptorPool = m_currentPool;
        return vkAllocateDescriptorSets(device, &info, pSet);
    default: 
        return res;
    }
}

VkDescriptorPool vkw::DescriptorAllocator::GetFreePool() 
{
    if (!m_freePools.empty()) {
        VkDescriptorPool p = m_freePools.back();
        m_freePools.pop_back();
        return p;
    }   
    else {
        return CreatePool();
    }
}

VkDescriptorPool vkw::DescriptorAllocator::CreatePool() 
{
    std::vector<VkDescriptorPoolSize> sizes;
    for (const auto& s : poolSizes) {
        sizes.push_back({ s.first, (uint32_t)(s.second * setsPerPool) });
    }
    auto info = vkw::init::DescriptorPoolCreateInfo(
        setsPerPool, sizes.size(), sizes.data());

    VkDescriptorPool pool;
    VK_CHECK(vkCreateDescriptorPool(device, &info, nullptr, &pool));
    return pool;
}