#include "vk_wrapper/shader.h"
#include <fstream>
#include <assert.h>
#include "vk_wrapper/initializers.h"
#include <string.h>
#include <sstream>
#include <array>
#include <stdlib.h>


vkw::ShaderCompiler::ShaderCompiler(
    vkw::Device* device, const std::string& file, Stage stage) 
{
    m_device = device;
    m_file = file;
    m_stage = stage;
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

std::vector<uint32_t> vkw::ShaderCompiler::ReadFileBinary(const std::string& path) 
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    assert(file.good() && file.is_open());

    size_t fileByteSize = file.tellg();
    file.seekg(0);

    //size_t bufSize = (fileByteSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    size_t bufSize = fileByteSize / sizeof(uint32_t);
    std::vector<uint32_t> buf(bufSize, 0);
    file.read((char*)buf.data(), fileByteSize);
    file.close();

    return buf;
}

void vkw::ShaderCompiler::WriteFile(const std::string& path, const std::string& contents) 
{
    std::ofstream file(path);
    assert(file.good());
    file << contents;
    file.close();
}

void vkw::ShaderCompiler::DeleteFile(const std::string& file) 
{
    if (remove(file.c_str())) {
        throw std::runtime_error("remove() failed on " + file);
    }    
}

void vkw::ShaderCompiler::DefineConstant(const std::string& name, const std::string& value) 
{
    m_constants[name] = value;    
}

std::string vkw::ShaderCompiler::Preprocess(
    const std::string& glslSource, 
    const std::vector<std::string>& includeDirs,
    const std::unordered_map<std::string, std::string>& constants) 
{
    std::stringstream input(glslSource);
    std::stringstream output;
    
    while (!input.eof()) {
        char line[glslSource.size()];
        input.getline(line, glslSource.size());

        // TODO : include the #included files.
        // TODO : replace the constant lines.

        output << line << "\n";
    }
    return output.str();
}

VkShaderModule vkw::ShaderCompiler::Compile(const std::vector<std::string>& includeDirs) 
{
    std::string ppSource = Preprocess(m_glslSource, includeDirs, m_constants);
    std::vector<uint32_t> spirv = CompileToSpirv(ppSource, m_stage, m_file);
    
    //auto info = vkw::init::ShaderModuleCreateInfo(
    //    spirv.size() * sizeof(uint32_t), spirv.data());

    VkShaderModule shader = 0;
    /*VkResult res = vkCreateShaderModule(m_device->logicalDevice, &info, nullptr, &shader);
    if (res != VK_SUCCESS) {
        printf("[-] Error creating shader in file %s\n", m_file.c_str());
        assert(false);
    }*/
    return shader;
}

std::vector<uint32_t> vkw::ShaderCompiler::CompileToSpirv(
    const std::string& glslSource, Stage stage,
    const std::string& fileName) 
{
    std::string glslTmpFile = fileName + ".pp";
    std::string spirvTmpFile = fileName + ".spv";

    std::string cmd = "glslangValidator -V " + glslTmpFile + " -o " + spirvTmpFile;
    // We redirect stderr to stdout.
    cmd += " 2>&1";
    // Specify the shader stage
    switch (stage) {
    case Stage::COMP: cmd += " -S comp"; break;
    case Stage::VERT: cmd += " -S vert"; break;
    case Stage::FRAG: cmd += " -S frag"; break;
    default: assert(false);
    }

    printf("[+] Glsl source =\n%s", glslSource.c_str());
    printf("[+] Compile cmd = %s\n", cmd.c_str());

    WriteFile(glslTmpFile, glslSource);  

    std::string output;
    int e = ExecuteCommand(cmd, output);  
    if (e) {
        printf("[-] Shader compilation error (%d) in file %s\n%s\n", 
            e, fileName.c_str(), output.c_str());
        DeleteFile(glslTmpFile);
        DeleteFile(spirvTmpFile);
        exit(-1);
    }
    std::vector<uint32_t> spirv = ReadFileBinary(spirvTmpFile);
    DeleteFile(glslTmpFile);
    DeleteFile(spirvTmpFile);
    return spirv;
}

int vkw::ShaderCompiler::ExecuteCommand(const std::string& cmd, std::string& output) 
{
    std::array<char, 128> buf;
    output = "";

    auto pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen() failed!");

    while (!feof(pipe)) {
        if (fgets(buf.data(), 128, pipe) != nullptr)
            output += buf.data();
    }
    return pclose(pipe);
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