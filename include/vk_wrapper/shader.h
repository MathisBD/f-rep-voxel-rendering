#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include "vk_wrapper/device.h"


namespace vkw
{
    // This 'compiler' will replace all lines in the glsl source file of the form :
    // #constant NAME
    // with lines :
    // #define NAME VALUE
    // where VALUE is a string that can be set with compiler.DefineConstant(NAME, VALUE);
    class ShaderCompiler
    {
    public:
        ShaderCompiler(vkw::Device* device, const std::string& file);
        void DefineConstant(const std::string& name, const std::string& value);
        VkShaderModule Compile(const std::vector<std::string>& includeDirs);
    private:
        // Maps from name to value.
        std::unordered_map<std::string, std::string> m_constants;

        vkw::Device* m_device;
        std::string m_file;
        std::string m_glslSource;
        
        static std::string ReadFile(const std::string& file);
        static std::string ReplaceConstants(
            const std::string& glslSource, 
            const std::unordered_map<std::string, std::string>& constants);
        static std::vector<uint32_t> CompileToSpirv(const std::string& glslSource);
    };


}