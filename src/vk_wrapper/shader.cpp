#include "vk_wrapper/shader.h"
#include <fstream>
#include <assert.h>
#include "vk_wrapper/initializers.h"


void vkw::Shader::Init(VkDevice dev_, const std::string& path) 
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
    printf("word count = %u\n", (uint32_t)(bufSize * sizeof(uint32_t)));

    VkResult res = vkCreateShaderModule(device, &info, nullptr, &shader);
    if (res != VK_SUCCESS) {
        printf("Error creating shader in file %s\n", path.c_str());
        assert(false);
    }
}

void vkw::Shader::Cleanup() 
{
    vkDestroyShaderModule(device, shader, nullptr);    
}