#pragma once
#include <vector>
#include <vulkan/vulkan.h>
#include <unordered_map>


namespace vkw
{
    class DescriptorAllocator
    {
    public:
        // The maximum number of descriptor sets per pool.
        uint32_t setsPerPool = 1024;
        // The maximum number of descriptors of each type per pool.
        // Theses sizes are multiplied by setsPerPool.
        std::vector<std::pair<VkDescriptorType, float>> poolSizes =
        {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,            1.0f },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,    1.0f },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,            1.0f },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,    1.0f },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,             1.0f }
        };
		
        VkDevice device = VK_NULL_HANDLE;

        void Init(VkDevice device);
        void Cleanup();

        VkResult Allocate(VkDescriptorSet* set, VkDescriptorSetLayout layout);
        void ResetPools();
    private:
        VkDescriptorPool m_currentPool = VK_NULL_HANDLE;
        std::vector<VkDescriptorPool> m_freePools;
        std::vector<VkDescriptorPool> m_usedPools;

        VkDescriptorPool GetFreePool();
        VkDescriptorPool CreatePool();
    };


    class DescriptorLayoutCache
    {
        public:
            VkDevice device;

            void Init(VkDevice device);
            void Cleanup();

            VkDescriptorSetLayout CreateDescriptorLayout(VkDescriptorSetLayoutCreateInfo* info);
        private:
            struct HashableDSLI
            {
                std::vector<VkDescriptorSetLayoutBinding> bindings;

                bool operator==(const HashableDSLI& other) const;
                size_t hash() const;
            };
            struct Hash
            {
                std::size_t operator()(const HashableDSLI& li) const {
                    return li.hash();
                }
            };

            std::unordered_map<HashableDSLI, VkDescriptorSetLayout, Hash> m_cache;
    };

    class DescriptorBuilder
    {
    public:
        DescriptorBuilder(
            DescriptorLayoutCache* cache, 
            DescriptorAllocator* allocator);

        DescriptorBuilder& BindBuffer(
            uint32_t binding,
            VkDescriptorBufferInfo* bufferInfo,
            VkDescriptorType type,
            VkShaderStageFlags stageFlags);
        DescriptorBuilder& BindImage(
            uint32_t binding,
            VkDescriptorImageInfo* imageInfo,
            VkDescriptorType type,
            VkShaderStageFlags stageFlags);
        // The set and/or the layout can be set to nullptr.
        void Build(VkDescriptorSet* set, VkDescriptorSetLayout* layout);
    private:
        DescriptorLayoutCache* m_cache;
        DescriptorAllocator* m_allocator;

        std::vector<VkDescriptorSetLayoutBinding> m_bindings;
        std::vector<VkWriteDescriptorSet> m_writes;
    };
}