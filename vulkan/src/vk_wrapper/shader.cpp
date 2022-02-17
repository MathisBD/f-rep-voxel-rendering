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
#include <unordered_set>


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

void vkw::ShaderCompiler::ClearConstants() 
{
    m_constants.clear();    
}

std::string vkw::ShaderCompiler::Preprocess(
    const std::string& glslSource, const std::string& file) 
{
    std::stringstream input(glslSource);
    std::stringstream output;
    
    std::string line;
    auto error = [&] (const std::string& msg) {
        std::stringstream e;
        e << "[-] Shader compile error in file " << file << "\n"; 
        e << "\t" << line << "\n";
        e << "\t--> " << msg << "\n";
        throw std::runtime_error(e.str());
    };

    std::unordered_set<std::string> definedConstants;
    while (!input.eof()) {
        getline(input, line);

        auto words = StringUtils::Split(line, {' ', '\t'});
        // #include
        if (words.size() > 0 && words[0] == "#include") {
            // Get the file name 
            if (words.size() == 1) {
                error("syntax error : unexpected end of line.");
            }
            if (words.size() > 2) {
                error("syntax error : expected a newline.");
            }
            std::string file = words[1];
            if(file.front() != '"' || file.back() != '"') {
                error("syntax error : expected a quote-enclosed file name.");
            }
            file = file.substr(1, file.size() - 2);
            file = m_shaderDir + file;
            
            // Insert the file contents
            try {
                std::string contents = ReadFile(file) + "\n";
                input.str(contents + input.str().substr(input.tellg()));
            } catch (...) {
                error("failed to read file " + file);
            }
        }
        // #constant
        else if (words.size() > 0 && words[0] == "#constant") {
            // Get the constant name
            if (words.size() == 1) {
                error("syntax error : unexpected end of line.");
            }
            if (words.size() > 2) {
                error("syntax error : expected a newline.");
            }
            std::string name = words[1];

            // Check for constant redefinition
            if (definedConstants.find(name) != definedConstants.end()) {
                error("redeclaration of constant " + name);
            }
            definedConstants.insert(name);
            
            // Get the constant value
            if (m_constants.find(name) == m_constants.end()) {
                error("the value of constant " + name + " has not been set.");
            }
            std::string value = m_constants[name];

            // Insert the #define macro
            std::string macro = "#define " + name + " " + value + "\n";
            input.str(macro + input.str().substr(input.tellg()));
        }
        else {
            output << line << "\n";
        }
    }
    return output.str();
}

VkShaderModule vkw::ShaderCompiler::Compile(const std::string& file, Stage stage) 
{
    std::string glslSource = ReadFile(m_shaderDir + file);
    std::string ppSource = Preprocess(glslSource, m_shaderDir + file);
    std::vector<uint32_t> spirv = CompileToSpirv(ppSource, stage, file);
    
    auto info = vkw::init::ShaderModuleCreateInfo(
        spirv.size() * sizeof(uint32_t), spirv.data());

    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(m_device->logicalDevice, &info, nullptr, &shader));
    m_device->NameObject(shader, "shader (compiled from " + m_shaderDir + file + ")");

    return shader;
}

std::vector<uint32_t> vkw::ShaderCompiler::CompileToSpirv(
    const std::string& glslSource, Stage stage,
    const std::string& fileName) 
{
    std::string glslTmpFile = m_shaderDir + fileName + ".pp";
    std::string spirvTmpFile = m_shaderDir + fileName + ".spv";

    std::string cmd = "glslangValidator -g -V " + glslTmpFile + " -o " + spirvTmpFile;
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
    //DeleteFile(glslTmpFile);
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
