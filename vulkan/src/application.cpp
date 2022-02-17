#include "application.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"
#include "utils/running_average.h"
#include "utils/timer.h"
#include "shapes.h"
#include "csg/expression.h"
#include "csg/tape.h"


Application::Application(Params params) : m_params(params), m_frameTime(32) 
{
    m_enableValidationLayers = params.enableValidationLayers;
    m_enableShaderDebugPrintf = params.enableShaderDebugPrintf;
}

void Application::Init()
{
    EngineBase::Init();

    m_voxels.gridDims = m_params.gridDims;
    m_voxels.lowVertex = { -20, -20, -20 };
    m_voxels.worldSize = 40;
    m_voxels.Init(&m_device, m_params.shape);
    m_cleanupQueue.AddFunction([=] { m_voxels.Cleanup(); });
    
    m_target.Init(&m_device, m_windowExtent, m_params.temporalSampleCount);
    ImmediateSubmit([=] (VkCommandBuffer cmd) { 
        m_target.ZeroOutImage(cmd); 
    });
    m_cleanupQueue.AddFunction([=] { m_target.Cleanup(); });

    m_voxelizer.Init(&m_device, &m_descAllocator, &m_descCache, &m_voxels);
    m_cleanupQueue.AddFunction([=] { m_voxelizer.Cleanup(); });  

    m_renderer.Init(
        &m_device, &m_descAllocator, &m_descCache,
        &m_target, m_windowExtent, m_surface);
    m_cleanupQueue.AddFunction([=] { m_renderer.Cleanup(); });
    // We have to signal the renderer's sem so that the first frame can work normally.
    m_renderer.SignalRenderSem();

    m_raytracer.Init(
        &m_device, &m_descAllocator, &m_descCache,
        &m_target, &m_voxels);
    m_cleanupQueue.AddFunction([=] { m_raytracer.Cleanup(); });
    m_raytracer.SetBackgroundColor({ 0.0f, 0.0f, 0.0f });  

    // Camera
    m_camera.position = { 0, 0, 40.0f };
    m_camera.forward = { 0, 0, -1 };
    m_camera.initialUp = { 0, 1, 0 };
    m_camera.fovDeg = 45;
    m_camera.aspectRatio = m_windowExtent.width / (float)m_windowExtent.height;
    m_camera.Init();

    PrintInfo();
}

void Application::Draw() 
{
    m_frameTime.AddSample(Timer::s_dt);
    if (m_params.printFPS) {
        printf("Frame time : %.1fms (%.1ffps)\n", 
            1000.0f * m_frameTime.GetAverage(), 1.0f / m_frameTime.GetAverage());
    }

    float time = 0.0f;
    if (m_params.voxelizeRealTime) {
        time = Timer::s_time;
    }

    // Voxelize
    VkSemaphore traceWaitSem;
    if (m_params.voxelizeRealTime || !m_voxelizedOnce) {
        m_voxelizer.Voxelize(m_renderer.GetRenderSem(), time);
        traceWaitSem = m_voxelizer.GetVoxelizeSem();
        m_voxelizedOnce = true;
    }
    else {
        traceWaitSem = m_renderer.GetRenderSem();    
    }
    //m_voxelizer.PrintStats();

    // Raytrace
    m_camera.Update(m_inputManager);
    m_raytracer.Trace(traceWaitSem, &m_camera, time);
    
    // Render
    m_renderer.BeginFrame();
    m_renderer.Render(m_raytracer.GetTraceSem());
    m_renderer.EndFrame();
}

void Application::PrintInfo() 
{
    if (m_params.printHardwareInfo) {
        printf("[+] Using device %s\n", m_device.properties.deviceName);
        printf("[+] Queue families :\n\tgraphics=%u\n\tcompute=%u\n\ttransfer=%u\n",
            m_device.queueFamilies.graphics, 
            m_device.queueFamilies.compute,
            m_device.queueFamilies.transfer);
        printf("[+] Shader invocation limits :\n");
        printf("\tmax work group size=(%u %u %u)\n", 
            m_device.properties.limits.maxComputeWorkGroupSize[0],
            m_device.properties.limits.maxComputeWorkGroupSize[1],
            m_device.properties.limits.maxComputeWorkGroupSize[2]);
        printf("\tmax work group count=(%u %u %u)\n", 
            m_device.properties.limits.maxComputeWorkGroupCount[0],
            m_device.properties.limits.maxComputeWorkGroupCount[1],
            m_device.properties.limits.maxComputeWorkGroupCount[2]);
        printf("\tmax invocation count=%u\n", m_device.properties.limits.maxComputeWorkGroupInvocations);
        printf("\tmax shared memory size=%u\n", m_device.properties.limits.maxComputeSharedMemorySize);
        
    }

    printf("[+] Grid dimensions: ");
    uint32_t totalDim = 1;
    for (uint32_t i = 0; i < m_voxels.gridLevels; i++) {
        printf("%u ", m_voxels.gridDims[i]);
        totalDim *= m_voxels.gridDims[i];
    }
    printf(" total=%u\n", totalDim);
}
