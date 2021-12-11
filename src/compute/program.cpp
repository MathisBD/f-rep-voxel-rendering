#include "compute/program.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <assert.h>


const std::string& Program::LoadSource() 
{
    std::ifstream file;
    file.open(m_sourcePath);
    assert(file.good());

    std::string line;
    std::stringstream ss;
    while (getline(file, line)) {
        ss << line << "\n";
    }
    return ss.str();
}

Program::Program(std::shared_ptr<cl::Context> ctx, const std::string& sourcePath) 
{
    m_sourcePath = sourcePath;
    m_ctx = ctx;
    const std::string& source = LoadSource();
    m_prg = std::make_unique<cl::Program>(*ctx, source);
}

/*void Program::Build() 
{
    try {
        m_prg->build(devices);
    }
    catch (cl::Error& e) {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            for (cl::Device dev : devices) {
                // Check the build status
                cl_build_status status = m_prg->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if (status != CL_BUILD_ERROR)
                    continue;

                // Get the build log
                std::string name     = dev.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << "Build log for " << name << ":" << std::endl
                        << buildlog << std::endl;
            }
        }
        else {
            throw e;
        }
    }    
}*/

const cl::Context& Program::GetContext() const {
    return *m_ctx;
}

const cl::Kernel& Program::GetKernel(const std::string& name) const {
    return cl::Kernel(*m_prg, name.c_str());    
}