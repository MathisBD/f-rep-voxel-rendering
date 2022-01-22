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
        enum class Stage {
            COMP,
            VERT,
            FRAG
        };

        ShaderCompiler(vkw::Device* device, const std::string& shaderDir);
        void DefineConstant(const std::string& name, const std::string& value);
        VkShaderModule Compile(const std::string& file, Stage stage);
    private:
        // Maps from name to value.
        std::unordered_map<std::string, std::string> m_constants;

        vkw::Device* m_device;
        std::string m_shaderDir;
        
        static std::string ReadFile(const std::string& file);
        static std::vector<uint32_t> ReadFileBinary(const std::string& file);
        static void WriteFile(const std::string& file, const std::string& contents);
        static void DeleteFile(const std::string& file);

        // Execute a shell command and returns the exit status code.
        static int ExecuteCommand(const std::string& cmd, std::string& output);

        std::string Preprocess(
            const std::string& glslSource,
            const std::unordered_map<std::string, std::string>& constants);
        std::vector<uint32_t> CompileToSpirv(
            const std::string& glslSource, Stage stage,
            const std::string& fileName);
    };


}