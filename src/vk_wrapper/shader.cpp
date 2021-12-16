#include "vk_wrapper/shader.h"
#include <fstream>
#include <assert.h>
#include "vk_wrapper/initializers.h"


void vkw::Shader::Init(VkDevice dev, const std::string& path) 
{
    this->device = dev;    

    // load the SPIRV source code
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    assert(file.good() && file.is_open());

    size_t fileByteSize = file.tellg();
    size_t bufSize = 1 + (fileByteSize / sizeof(uint32_t));
    uint32_t* buf = new uint32_t[bufSize];

    file.seekg(0);
    file.read((char*)buf, fileByteSize);
    file.close();

    // create the shader
    auto info = vkw::init::ShaderModuleCreateInfo(
        bufSize * sizeof(uint32_t), buf);
    VkResult res = vkCreateShaderModule(device, &info, nullptr, &shader);
    if (res != VK_SUCCESS) {
        printf("Error creating shader in file %s\n", path);
        assert(false);
    }
}

void vkw::Shader::Cleanup() 
{
    vkDestroyShaderModule(device, shader, nullptr);    
}