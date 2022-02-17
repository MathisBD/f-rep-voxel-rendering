#pragma once
#include <vector>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include <type_traits>


namespace vkw
{
    class SpecConsts
    {
    public:
        template <typename T>
        void AddEntry(uint32_t id, T value)
        {
            VkSpecializationMapEntry entry = {};
            entry.constantID = id;
            entry.size = sizeof(T);
            entry.offset = m_data.size();
            m_entries.push_back(entry);

            for (size_t i = 0; i < entry.size; i++) {
                m_data.push_back(0);
            }    
            memcpy(m_data.data() + entry.offset, &value, entry.size);
        }

        void Build()
        {
            m_info = {
                (uint32_t)m_entries.size(),
                m_entries.data(),
                (uint32_t)m_data.size(),
                m_data.data()
            };
        }

        VkSpecializationInfo* GetInfo()
        {
            return &m_info;
        }
    private:
        std::vector<uint8_t> m_data;
        std::vector<VkSpecializationMapEntry> m_entries;
        VkSpecializationInfo m_info;
    };
}