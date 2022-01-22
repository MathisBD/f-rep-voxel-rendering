#include "vk_wrapper/shader.h"
#include <fstream>
#include <assert.h>
#include "vk_wrapper/initializers.h"
#include <string.h>
#include <sstream>
#include <array>
#include <stdlib.h>
#include "utils/string_utils.h"
#include "vk_wrapper/vk_check.h"


vkw::ShaderCompiler::ShaderCompiler(
    vkw::Device* device, const std::string& shaderDir) 
{
    m_device = device;
    m_shaderDir = shaderDir;
}

std::string vkw::ShaderCompiler::ReadFile(const std::string& path) 
{
    std::ifstream file(path, std::ios::ate);
    if(!file.good() || !file.is_open()) {
        throw std::runtime_error("could not open file " + path);
    }

    size_t fileByteSize = file.tellg();
    file.seekg(0);

    char buf[fileByteSize+1];
    file.read(buf, fileByteSize);
    file.close();
    buf[fileByteSize] = 0;

    return std::string(buf);
}

std::vector<uint32_t> vkw::ShaderCompiler::ReadFileBinary(const std::string& path) 
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if(!file.good() || !file.is_open()) {
        throw std::runtime_error("could not open file " + path);
    }
    size_t fileByteSize = file.tellg();
    file.seekg(0);

    size_t bufSize = (fileByteSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    //size_t bufSize = fileByteSize / sizeof(uint32_t);
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
    const std::unordered_map<std::string, std::string>& constants) 
{
    std::stringstream input(glslSource);
    std::stringstream output;
    
    while (!input.eof()) {
        std::string line;
        getline(input, line);

        auto words = StringUtils::Split(line, {' ', '\n', '\t'});
        // #include
        if (words.size() > 0 && words[0] == "#include") {
            assert(words.size() == 2);
            std::string file = words[1];
            assert(file.front() == '"' && file.back() == '"');
            file = file.substr(1, file.size() - 2);
            file = m_shaderDir + file;
            
            input.str(ReadFile(file) + "\n" + input.str().substr(input.tellg()));
        }
        // #constant
        /*else if (words.size() > 0 && words[0] == "#constant") {

        }*/
        else {
            output << line << "\n";
        }
    }
    return output.str();
}

VkShaderModule vkw::ShaderCompiler::Compile(const std::string& file, Stage stage) 
{
    std::string glslSource = ReadFile(m_shaderDir + file);
    std::string ppSource = Preprocess(glslSource, m_constants);
    std::vector<uint32_t> spirv = CompileToSpirv(ppSource, stage, file);
    
    auto info = vkw::init::ShaderModuleCreateInfo(
        spirv.size() * sizeof(uint32_t), spirv.data());

    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(m_device->logicalDevice, &info, nullptr, &shader));
    return shader;
}

std::vector<uint32_t> vkw::ShaderCompiler::CompileToSpirv(
    const std::string& glslSource, Stage stage,
    const std::string& fileName) 
{
    std::string glslTmpFile = m_shaderDir + fileName + ".pp";
    std::string spirvTmpFile = m_shaderDir + fileName + ".spv";

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

    WriteFile(glslTmpFile, glslSource);  

    std::string output;
    int e = ExecuteCommand(cmd, output);  
    if (e) {
        printf("[-] Shader compilation error (%d) in file %s\n\t%s\n", 
            e, (m_shaderDir + fileName).c_str(), 
            StringUtils::Replace(output, "\n", "\n\t").c_str());
        //DeleteFile(glslTmpFile);
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