#include "application.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"
#include "utils/running_average.h"
#include "utils/timer.h"
#include "shapes.h"
#include "engine/csg_expression.h"
#include "engine/csg_tape.h"


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
    SetupScene();

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
    m_voxels.gridDims = { 4, 4, 4 };
    m_voxels.lowVertex = { -20, -20, -20 };
    m_voxels.worldSize = 40;

    m_voxels.Init(m_vmaAllocator);

    m_cleanupQueue.AddFunction([=] {
        m_voxels.nodeBuffer.Cleanup();
        m_voxels.childBuffer.Cleanup();
        m_voxels.voxelBuffer.Cleanup();
        m_voxels.tapeBuffer.Cleanup();
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

void Application::SetupScene() 
{
    // Build the shape expression and tape
    //csg::Expr sphere = Shapes::Sphere({10, 0, 0}, 20.0f);
    csg::Expr shape = Shapes::TangleCube({0, 0, 0}, 4);
    m_builder.Init(m_vmaAllocator, &m_voxels, shape);
    m_voxels.tape.Print();

    auto start = std::chrono::high_resolution_clock::now();
    // Create the voxels
    m_builder.BuildScene();

    auto build = std::chrono::high_resolution_clock::now();
    printf("[+] Build time = %ldms\n", 
        std::chrono::duration_cast<std::chrono::milliseconds>(build - start).count());

    ImmediateSubmit([=] (VkCommandBuffer cmd) { 
        m_builder.UploadSceneToGPU(cmd);
    });
    auto upload = std::chrono::high_resolution_clock::now();
    printf("[+] Upload time = %ldms\n", 
        std::chrono::duration_cast<std::chrono::milliseconds>(upload - build).count());

    // We can now cleanup the scene builder (and delete its staging buffers)
    m_builder.Cleanup();
}


void Application::Draw() 
{
    m_frameTime.AddSample(Timer::s_dt);
    //printf("Frame time : %.1fms (%.1ffps)\n", 
    //    1000.0f * m_frameTime.GetAverage(), 1.0f / m_frameTime.GetAverage());

    m_camera.Update(m_inputManager);
    m_raytracer.Trace(m_renderer.GetRenderSemaphore(), &m_camera);
    
    m_renderer.BeginFrame();
    m_renderer.Render(m_raytracer.GetComputeSemaphore());
    m_renderer.EndFrame();
}
