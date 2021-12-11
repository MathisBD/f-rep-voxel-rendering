

#include <CL/cl2.hpp>
#include "compute/device.h"
#include <iostream>
#include <assert.h>
#include <chrono>
#include <stdio.h>


std::string kernelSource = "\
__kernel void vadd(\
    __global float* a,\
    __global float* b,\
    __global float* c,\
    int count)\
{\
    int id = get_global_id(0);\
    c[id] = a[id] + b[id];\
}";


void VectorAdd(std::vector<float>& h_a, std::vector<float>& h_b, std::vector<float>& h_c)
{
    size_t count = h_a.size();
    assert(count == h_b.size() && count == h_c.size());

    

    cl::Context ctx(CL_DEVICE_TYPE_ALL,;
    std::vector<cl::Device> devices;
    ctx.getInfo(CL_CONTEXT_DEVICES, &devices);
    
    printf("Context devices :\n");
    for (const cl::Device& d : devices) {
        auto dev = Device(std::make_unique<cl::Device>(d));
        dev.Print();
    }
    
    cl::Program prg(ctx, kernelSource);
    prg.build();

    cl::CommandQueue queue(ctx); 
    
    cl::Buffer d_a(ctx, h_a.begin(), h_a.end(), true);
    cl::Buffer d_b(ctx, h_b.begin(), h_b.end(), true);
    cl::Buffer d_c(ctx, CL_MEM_WRITE_ONLY, count * sizeof(float));

    auto vadd = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int>(cl::Kernel(prg, "vadd"));
    vadd(cl::EnqueueArgs(queue, cl::NDRange(count)),
        d_a, d_b, d_c, count);
    cl::copy(queue, d_c, h_c.begin(), h_c.end());
}


int main()
{
    size_t count = 1000000;
    std::vector<float> h_a(count);
    std::vector<float> h_b(count);
    std::vector<float> h_c(count);
    for (size_t i = 0; i < count; i++) {
        h_a[i] = 0.4f * i;
        h_b[i] = 0.1f * i;   
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();
        VectorAdd(h_a, h_b, h_c);
        auto end = std::chrono::high_resolution_clock::now();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Addition completed in %lums\n", millis.count());
    }
    catch (cl::Error err)
    {
        std::cout << "OpenCL Error: " << err.what() << " returned " << err.err() << std::endl;
        std::cout << "Check cl.h for error codes." << std::endl;
        throw err;
    }
    for (size_t i = 0; i < count; i++) {
        assert(h_c[i] == h_a[i] + h_b[i]);   
    }
    return 0;
}