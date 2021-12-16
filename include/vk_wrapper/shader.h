#pragma once
#include <vulkan/vulkan.h>
#include <string>



namespace vkw
{
    class Shader
    {
    public:
        VkDevice device;
        VkShaderModule shader;

        void Init(VkDevice dev, const std::string& path);
        void Cleanup();
    };


}