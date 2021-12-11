#pragma once
#include <CL/cl2.hpp>
#include <string.h>


class Device
{
public:
    Device(const cl::Device& dev); 

    const cl::Device& GetCLDevice() const;
    void Print() const;

    const std::string& GetName() const;
    const std::string& GetOpenCLVersion() const;
    int GetComputeUnits() const;
    size_t GetLocalMemSize() const;
    size_t GetGlobalMemSize() const;
    size_t GetMaxWorkGroupSize() const;
    const std::vector<size_t>& GetMaxWorkItemSizes() const;
private:
    cl::Device m_dev;
    std::string m_name;
    std::string m_openCLVersion;
    int m_computeUnits;
    size_t m_localMemSize;
    size_t m_globalMemSize;
    size_t m_maxWorkGroupSize;
    std::vector<size_t> m_maxWorkItemSizes; 
};


