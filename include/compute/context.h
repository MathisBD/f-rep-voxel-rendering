#pragma once
#include "compute/device.h"


class Context
{
public:
    Context(std::shared_ptr<Device> dev);

    const cl::Context& GetCLContext() const;
    const Device& GetDevice() const;
private:
    cl::Context m_ctx;
    std::shared_ptr<Device> m_dev;
};