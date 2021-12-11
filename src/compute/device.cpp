#include "compute/device.h"


Device::Device(const cl::Device& dev)
{
    m_dev = dev;
    m_dev.getInfo(CL_DEVICE_NAME, &m_name);
    m_dev.getInfo(CL_DEVICE_OPENCL_C_VERSION, &m_openCLVersion);
    m_dev.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &m_computeUnits);
    m_dev.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &m_localMemSize);
    m_dev.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &m_globalMemSize);
    m_dev.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &m_maxWorkGroupSize);
    m_dev.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &m_maxWorkItemSizes);
}

const std::string& Device::GetName() const {
    return m_name;
}

const std::string& Device::GetOpenCLVersion() const {
    return m_openCLVersion;
}

int Device::GetComputeUnits() const {
    return m_computeUnits;
}

size_t Device::GetGlobalMemSize() const {
    return m_globalMemSize;
}

size_t Device::GetLocalMemSize() const {
    return m_localMemSize;
}

size_t Device::GetMaxWorkGroupSize() const {
    return m_maxWorkGroupSize;
}

const std::vector<size_t>& Device::GetMaxWorkItemSizes() const {
    return m_maxWorkItemSizes;
}

void Device::Print() const 
{
    printf("%s (%s):\n", m_name.c_str(), m_openCLVersion.c_str());
    printf("\tcompute units=%d\n", m_computeUnits);
    printf("\tlocal/global mem=%lu/%lu\n", m_localMemSize, m_globalMemSize);
    printf("\twork group size : %lu (", m_maxWorkGroupSize);
    for (size_t s : m_maxWorkItemSizes) {
        printf(" %lu", s);
    }
    printf(" )\n");
}