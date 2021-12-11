#pragma once
#include <CL/cl2.hpp>


class Platform
{
public:
    static const std::vector<Platform>& AllPlatforms();;

    const std::string& GetName() const { return m_name; };
    const std::string& GetVendor() const { return m_vendor; };
    const std::string& GetVersion() const { return m_version; };
    const std::vector<Device> GetDevices() const { return m_devices; };
private:
    cl::Platform m_plat;
    std::string m_name;
    std::string m_vendor;
    std::string m_version;
    std::vector<Device> m_devices;

    static std::once_flag s_collectedAllPlatforms;

    Platform(const cl::Platform& plat);
};