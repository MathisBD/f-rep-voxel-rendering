#include "engine/raytracer.h"
#include "vk_wrapper/vk_check.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/shader.h"


void Raytracer::Init(
    vkw::Device* device, RenderTarget* target, VmaAllocator vmaAllocator,
    vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache) 
{
    m_device = device;
    m_target = target;
    m_vmaAllocator = vmaAllocator;
    m_descAllocator = descAllocator;
    m_descCache = descCache;

    InitCommands();
    InitSynchronization();
    InitBuffers();
    InitPipeline();

    // We only upload the voxels once.
    UpdateDDAVoxels();
}


void Raytracer::InitCommands() 
{
    // Queue
    vkGetDeviceQueue(m_device->logicalDevice, 
        m_device->queueFamilies.compute, 0, &m_queue);
    
    // Command pool
    auto poolInfo = vkw::init::CommandPoolCreateInfo(
        m_device->queueFamilies.compute);
    VK_CHECK(vkCreateCommandPool(m_device->logicalDevice, &poolInfo, nullptr, &m_cmdPool));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyCommandPool(m_device->logicalDevice, m_cmdPool, nullptr); 
    });
}

void Raytracer::InitSynchronization()
{
    // Fence
    auto fenceInfo = vkw::init::FenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    VK_CHECK(vkCreateFence(m_device->logicalDevice, &fenceInfo, nullptr, &m_fence));
    m_cleanupQueue.AddFunction([=] {
        vkDestroyFence(m_device->logicalDevice, m_fence, nullptr);
    })

    // Semaphore    
    auto semInfo = vkw::init::SemaphoreCreateInfo();
    VK_CHECK(vkCreateSemaphore(m_device->logicalDevice, &semInfo, nullptr, &m_semaphore));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroySemaphore(m_device->logicalDevice, m_semaphore, nullptr);
    });
}

void Raytracer::InitBuffers()
{
    // Uniform buffer
    m_ddaUniforms.Init(m_vmaAllocator);
    m_ddaUniforms.Allocate(
        sizeof(DDAUniforms), 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] { m_ddaUniforms.Cleanup(); });

    // Lights buffer
    m_ddaLights.Init(m_vmaAllocator);
    m_ddaLights.Allocate(
        sizeof(glm::vec4) + m_lightCount * sizeof(DDALight),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] { m_ddaLights.Cleanup(); });

    // Voxel grid
    m_voxelGrid = CubeGrid(128, { -10, -10, -10 }, 20);

    // Voxels buffer
    size_t dim = m_voxelGrid.dim;
    m_ddaVoxels.Init(m_vmaAllocator);
    m_ddaVoxels.Allocate(
        dim * dim * dim * sizeof(DDAVoxel),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] { m_ddaVoxels.Cleanup(); });

    // Camera
    m_camera.position = { 0, 0, 40.0f };
    m_camera.forward = { 0, 0, -1 };
    m_camera.initialUp = { 0, 1, 0 };
    m_camera.fovDeg = 45;
    m_camera.aspectRatio = m_windowExtent.width / (float)m_windowExtent.height;
    m_camera.Init();
}

void Application::InitPipeline() 
{
    // Load the shader
    vkw::Shader shader;
    shader.Init(m_device.logicalDevice, "../shaders/dda.comp.spv");
    
    // Descriptor Sets
    m_compute.dSets = std::vector<VkDescriptorSet>(1);
    auto dSetLayouts = std::vector<VkDescriptorSetLayout>(1);

    // Descriptor Set 0    
    auto outImageInfo = vkw::init::DescriptorImageInfo(
        m_target->sampler, m_target->view, VK_IMAGE_LAYOUT_GENERAL);
    auto ddaUniformsInfo = vkw::init::DescriptorBufferInfo(
        m_ddaUniforms.buffer, 0, m_ddaUniforms.size);
    auto ddaVoxelsInfo = vkw::init::DescriptorBufferInfo(
        m_ddaVoxels.buffer, 0, m_ddaVoxels.size);
    auto ddaLightsInfo = vkw::init::DescriptorBufferInfo(
        m_ddaLights.buffer, 0, m_ddaLights.size);
    vkw::DescriptorBuilder(&m_descCache, &m_descAllocator)
        .BindImage(0, &outImageInfo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(1, &ddaUniformsInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(2, &ddaVoxelsInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .BindBuffer(3, &ddaLightsInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .Build(&m_descSets[0], &dSetLayouts[0]);

    // Pipeline layout
    auto layoutInfo = vkw::init::PipelineLayoutCreateInfo();
    layoutInfo.setLayoutCount = (uint32_t)dSetLayouts.size();
    layoutInfo.pSetLayouts = dSetLayouts.data();
    VK_CHECK(vkCreatePipelineLayout(m_device->logicalDevice, &layoutInfo, nullptr, &m_pipelineLayout));
 
    // Pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = nullptr;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.stage = vkw::init::PipelineShaderStageCreateInfo(
        VK_SHADER_STAGE_COMPUTE_BIT, shader.shader);
    VK_CHECK(vkCreateComputePipelines(m_device->logicalDevice, VK_NULL_HANDLE, 
        1, &pipelineInfo, nullptr, &m_pipeline));

    // We can destroy the shader right away.
    shader.Cleanup();
    m_cleanupQueue.AddFunction([=] {
        vkDestroyPipelineLayout(m_device->logicalDevice, m_pipelineLayout, nullptr);
        vkDestroyPipeline(m_device->logicalDevice, m_pipeline, nullptr);
    });
}
