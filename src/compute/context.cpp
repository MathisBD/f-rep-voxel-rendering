#include "compute/context.h"


Context::Context(const Device& dev) :
    m_dev(dev),
    m_ctx(dev.GetCLDevice())
{
}

const cl::Context& Context::GetCLContext() const {
    return m_ctx;
}

const Device& Context::GetDevice() const {
    return m_dev;
}