#include "application.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"
#include "utils/running_average.h"
#include "utils/timer.h"


Application::Application() : m_frameTime(32) {}

void Application::Init(bool enableValidationLayers)
{
    EngineBase::Init(enableValidationLayers);

    printf("[+] Queue families :\n\tgraphics=%u\n\tcompute=%u\n\ttransfer=%u\n",
        m_device.queueFamilies.graphics, 
        m_device.queueFamilies.compute,
        m_device.queueFamilies.transfer);

    InitRenderTarget();
    InitVoxels();

    m_builder.Init(m_vmaAllocator, &m_voxels);
    m_cleanupQueue.AddFunction([=] { m_builder.Cleanup(); });
    
    // Create the voxels
    /*auto sphere = [] (float x, float y, float z) {
        glm::vec3 pos = { x, y, z };
        glm::vec3 center = { 0, 0, 0 };
        float radius = 15;
        return radius * radius - glm::length2(pos - center);
    };*/
    auto tanglecube = [] (float x, float y, float z) {
        x /= 3;
        y /= 3;
        z /= 3;
        float x2 = x*x;
        float y2 = y*y;
        float z2 = z*z;
        float x4 = x2*x2;
        float y4 = y2*y2;
        float z4 = z2*z2;
        return -(x4 + y4 + z4 - 8 * (x2 + y2 + z2) + 25);
    };
    /*auto barth_sextic = [] (float x, float y, float z) {
        auto square = [] (float a) { return a*a; };
        x /= 4;
        y /= 4;
        z /= 4;
        
        float t = (1 + glm::sqrt(5)) / 2;
        float x2 = x*x;
        float y2 = y*y;
        float z2 = z*z;
        float t2 = t*t;
        float res = 4 * (t2*x2 - y2) * (t2*y2 - z2) * (t2*z2 - x2) -
            (1 + 2*t) * square(x2 + y2 + z2 - 1);
        return res;    
    };*/
    m_builder.CreateVoxels(tanglecube);
    m_builder.AllocateGPUBuffers();
    ImmediateSubmit([=] (VkCommandBuffer cmd) { 
        m_builder.CopyStagingBuffers(cmd);
    });

    m_renderer.Init(
        &m_device, &m_descAllocator, &m_descCache,
        &m_target, m_windowExtent, m_surface);
    m_cleanupQueue.AddFunction([=] { m_renderer.Cleanup(); });

    m_raytracer.Init(
        &m_device, &m_descAllocator, &m_descCache,
        &m_target, &m_voxels, m_vmaAllocator);
    m_cleanupQueue.AddFunction([=] { m_raytracer.Cleanup(); });
    m_raytracer.SetBackgroundColor({ 0.0f, 0.0f, 0.0f });

    // Camera
    m_camera.position = { 0, 0, 40.0f };
    m_camera.forward = { 0, 0, -1 };
    m_camera.initialUp = { 0, 1, 0 };
    m_camera.fovDeg = 45;
    m_camera.aspectRatio = m_windowExtent.width / (float)m_windowExtent.height;
    m_camera.Init();
}

void Application::InitVoxels() 
{
    m_voxels.gridLevels = 1;
    m_voxels.gridDims = { 32 };
    m_voxels.lowVertex = { -10, -10, -10 };
    m_voxels.worldSize = 20;

    m_voxels.nodeBuffer.Init(m_vmaAllocator);
    m_voxels.childBuffer.Init(m_vmaAllocator);
    m_voxels.voxelBuffer.Init(m_vmaAllocator);

    m_cleanupQueue.AddFunction([=] {
        m_voxels.nodeBuffer.Cleanup();
        m_voxels.childBuffer.Cleanup();
        m_voxels.voxelBuffer.Cleanup();
    });
}

void Application::InitRenderTarget() 
{
    // Allocate the image
    m_target.image.Init(m_vmaAllocator);
    std::vector<uint32_t> queueFamilies;
    VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (m_device.queueFamilies.graphics != m_device.queueFamilies.compute) {
        queueFamilies = { 
            m_device.queueFamilies.graphics,
            m_device.queueFamilies.compute };
        sharingMode = VK_SHARING_MODE_CONCURRENT;
    }
    m_target.image.Allocate(
        m_windowExtent,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        sharingMode,
        &queueFamilies);
    m_cleanupQueue.AddFunction([=] { m_target.image.Cleanup(); });

    // Set the image layout (GENERAL).
    ImmediateSubmit(
        [=] (VkCommandBuffer cmd) { 
            m_target.image.ChangeLayout(cmd, 
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, VK_ACCESS_SHADER_WRITE_BIT);
        });

    // Image view
    auto viewInfo = vkw::init::ImageViewCreateInfo(
        m_target.image.format, m_target.image.image, VK_IMAGE_ASPECT_COLOR_BIT);
    VK_CHECK(vkCreateImageView(m_device.logicalDevice, &viewInfo, nullptr, &m_target.view));
    m_cleanupQueue.AddFunction([=] {
        vkDestroyImageView(m_device.logicalDevice, m_target.view, nullptr);
    });

    // Sampler
    auto samplerInfo = vkw::init::SamplerCreateInfo(VK_FILTER_NEAREST);
    VK_CHECK(vkCreateSampler(
        m_device.logicalDevice, &samplerInfo, nullptr, &m_target.sampler));
    m_cleanupQueue.AddFunction([=] {
        vkDestroySampler(m_device.logicalDevice, m_target.sampler, nullptr);
    });
}


void Application::Draw() 
{
    m_frameTime.AddSample(Timer::s_dt);
    printf("Frame time : %.1fms (%.1ffps)\n", 
        1000.0f * m_frameTime.GetAverage(), 1.0f / m_frameTime.GetAverage());

    m_camera.Update(m_inputManager);
    m_raytracer.Trace(m_renderer.GetRenderSemaphore(), &m_camera);
    
    m_renderer.BeginFrame();
    m_renderer.Render(m_raytracer.GetComputeSemaphore());
    m_renderer.EndFrame();
}

