#include "vk_wrapper/descriptor.h"
#include <algorithm>
#include <assert.h>



void vkw::DescriptorLayoutCache::Init(VkDevice dev)
{
    this->device = dev;
}

void vkw::DescriptorLayoutCache::Cleanup()
{
    for (const auto& entry : m_cache) {
        vkDestroyDescriptorSetLayout(device, entry.second, nullptr);
    }
}

VkDescriptorSetLayout vkw::DescriptorLayoutCache::CreateDescriptorLayout(
    VkDescriptorSetLayoutCreateInfo* info)
{
    // Create the HashableDSLI
    HashableDSLI key = {};
    key.bindings.reserve(info->bindingCount);
    for (uint32_t i = 0; i < info->bindingCount; i++) {
        key.bindings.push_back(info->pBindings[i]);
    }
    std::sort(
        key.bindings.begin(), 
        key.bindings.end(),
        [](const VkDescriptorSetLayoutBinding& a,
           const VkDescriptorSetLayoutBinding& b) {
               return a.binding < b.binding; 
        });
    // check there are no duplicate bindings
    for (size_t i = 0; i < key.bindings.size()-1; i++) {
        assert(key.bindings[i].binding < key.bindings[i+1].binding);
    }

    // Is it in the cache ?
    auto it = m_cache.find(key);
    if (it != m_cache.end()) {
        return it->second;
    }

    // Create a new layout
    VkDescriptorSetLayout layout;
    vkCreateDescriptorSetLayout(device, info, nullptr, &layout);
    m_cache[key] = layout;
    return layout;
}

bool vkw::DescriptorLayoutCache::HashableDSLI::operator==(
    const vkw::DescriptorLayoutCache::HashableDSLI& other) const
{
    if (bindings.size() != other.bindings.size()) {
        return false;
    }
    for (size_t i = 0; i < bindings.size(); i++) {
        const auto& b = bindings[i];
        const auto& ob = other.bindings[i];
        
        if (b.binding != ob.binding ||
            b.descriptorCount != ob.descriptorCount ||
            b.descriptorType != ob.descriptorType ||
            b.stageFlags != ob.stageFlags) {
            return false;
        }
    }
    return true;
}

size_t vkw::DescriptorLayoutCache::HashableDSLI::hash() const
{
    // Use a method similar to boost::hash_combine.
    auto hash = [](size_t seed, size_t val) {
        size_t tmp = std::hash<size_t>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed ^ tmp;
    };

    size_t seed = bindings.size();
    for (const auto& b : bindings) {
        seed = hash(seed, b.binding);
        seed = hash(seed, b.descriptorCount);
        seed = hash(seed, b.descriptorType);
        seed = hash(seed, b.stageFlags);
    }
    return seed;
}
            
                