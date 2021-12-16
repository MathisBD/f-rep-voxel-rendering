#include "vk_wrapper/descriptor.h"
#include <assert.h>
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/initializers.h"


vkw::DescriptorBuilder::DescriptorBuilder(
    DescriptorLayoutCache* cache, 
    DescriptorAllocator* allocator)
{
    assert(cache);
    assert(allocator);
    assert(cache->device == allocator->device);

    m_cache = cache;
    m_allocator = allocator;
}


vkw::DescriptorBuilder& vkw::DescriptorBuilder::BindBuffer(
    uint32_t b,
    VkDescriptorBufferInfo* bufferInfo,
    VkDescriptorType type,
    VkShaderStageFlags stageFlags)
{
    // add the binding
    VkDescriptorSetLayoutBinding binding = {};
    binding.descriptorCount = 1;
    binding.descriptorType = type;
    binding.binding = b;
    binding.pImmutableSamplers = nullptr;
    binding.stageFlags = stageFlags;
    m_bindings.push_back(binding);

    // add the binding write
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.pNext = nullptr;

    write.descriptorCount = 1;
    write.descriptorType = type;
    write.dstBinding = b;
    write.pBufferInfo = bufferInfo;
    m_writes.push_back(write);

    return *this;
}

vkw::DescriptorBuilder& vkw::DescriptorBuilder::BindImage(
    uint32_t b,
    VkDescriptorImageInfo* imageInfo,
    VkDescriptorType type,
    VkShaderStageFlags stageFlags)
{
    // add the binding
    VkDescriptorSetLayoutBinding binding = {};
    binding.descriptorCount = 1;
    binding.descriptorType = type;
    binding.binding = b;
    binding.pImmutableSamplers = nullptr;
    binding.stageFlags = stageFlags;
    m_bindings.push_back(binding);

    // add the binding write
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.pNext = nullptr;

    write.descriptorCount = 1;
    write.descriptorType = type;
    write.dstBinding = b;
    write.pImageInfo = imageInfo;
    m_writes.push_back(write);

    return *this;
}
        
void vkw::DescriptorBuilder::Build(VkDescriptorSet* pSet, VkDescriptorSetLayout* pLayout)
{
    // Use a local layout handle if the caller didn't provide any.
    VkDescriptorSetLayout localLayout;
    if (pLayout == nullptr) {
        pLayout = &localLayout;
    }

    // Create the layout
    auto layoutInfo = vkw::init::DescriptorSetLayoutCreateInfo(
        m_bindings.size(), m_bindings.data());
    *pLayout = m_cache->CreateDescriptorLayout(&layoutInfo);

    // The caller requested the layout only.
    if (pSet == nullptr) {
        return;
    }

    // Allocate the descriptor set.
    VK_CHECK(m_allocator->Allocate(pSet, *pLayout));

    // Make it point to the right buffers/images.
    for (size_t i = 0; i < m_writes.size(); i++) {
        m_writes[i].dstSet = *pSet;
    }
    vkUpdateDescriptorSets(m_allocator->device, (uint32_t)(m_writes.size()), m_writes.data(), 0, nullptr);
}

    