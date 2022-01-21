#include "vk_wrapper/shader.h"
#include <fstream>
#include <assert.h>
#include "vk_wrapper/initializers.h"
#include <string.h>
#include <sstream>


vkw::ShaderCompiler::ShaderCompiler(vkw::Device* device, const std::string& file) 
{
    m_device = device;
    m_file = file;
    m_glslSource = ReadFile(m_file);
}

std::string vkw::ShaderCompiler::ReadFile(const std::string& path) 
{
    std::ifstream file(path, std::ios::ate);
    assert(file.good() && file.is_open());

    size_t fileByteSize = file.tellg();
    file.seekg(0);

    char buf[fileByteSize];
    file.read(buf, fileByteSize);
    file.close();

    return std::string(buf);
}

void vkw::ShaderCompiler::DefineConstant(const std::string& name, const std::string& value) 
{
    m_constants[name] = value;    
}

std::string vkw::ShaderCompiler::ReplaceConstants(
    const std::string& glslSource, 
    const std::unordered_map<std::string, std::string>& constants) 
{
    std::stringstream input(glslSource);
    std::stringstream output;
    
    while (!input.eof()) {
        char line[glslSource.size()];
        input.getline(line, glslSource.size());

        // TODO : replace the constant lines.

        output << line;
    }
    return output.str();
}

VkShaderModule vkw::ShaderCompiler::Compile(const std::vector<std::string>& includeDirs) 
{
    std::string source = ReplaceConstants(m_glslSource, m_constants);
    std::vector<uint32_t> spirv = CompileToSpirv(source);
    
    auto info = vkw::init::ShaderModuleCreateInfo(
        spirv.size() * sizeof(uint32_t), spirv.data());

    VkShaderModule shader;
    VkResult res = vkCreateShaderModule(m_device->logicalDevice, &info, nullptr, &shader);
    if (res != VK_SUCCESS) {
        printf("[-] Error creating shader in file %s\n", m_file.c_str());
        assert(false);
    }
    return shader;
}

/*void vkw::Shader::Init(VkDevice dev_, const std::string& path) 
{
    device = dev_;    

    // load the SPIRV source code
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    assert(file.good() && file.is_open());

    size_t fileByteSize = file.tellg();
    size_t bufSize = fileByteSize / sizeof(uint32_t);
    uint32_t* buf = new uint32_t[bufSize];

    file.seekg(0);
    file.read((char*)buf, fileByteSize);
    file.close();

    // create the shader
    auto info = vkw::init::ShaderModuleCreateInfo(
        (uint32_t)(bufSize * sizeof(uint32_t)), buf);

    VkResult res = vkCreateShaderModule(device, &info, nullptr, &shader);
    if (res != VK_SUCCESS) {
        printf("Error creating shader in file %s\n", path.c_str());
        assert(false);
    }
}

void vkw::Shader::Cleanup() 
{
    vkDestroyShaderModule(device, shader, nullptr);    
}*/