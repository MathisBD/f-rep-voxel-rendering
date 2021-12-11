#pragma once
#include <CL/cl2.hpp>
#include "compute/context.h"


class Program
{
public:
    Program(std::shared_ptr<Context> ctx, const std::string& sourcePath);
    void Build();
    
    const cl::Context& GetContext() const;
    const cl::Kernel& GetKernel(const std::string& name) const;
private:
    std::string m_sourcePath;
    std::shared_ptr<cl::Context> m_ctx;
    cl::Program m_prg;
    
    const std::string& LoadSource();
};