#include "engine/engine_base.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include "VkBootstrap.h"
#include "vk_wrapper/shader.h"
#include "vk_wrapper/pipeline_builder.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"



void EngineBase::Init() 
{
    InitSDL();
    InitVulkanCore();
    InitVma();

    m_descriptorAllocator.Init(m_device.logicalDevice);
    m_cleanupQueue.AddFunction([=] { m_descriptorAllocator.Cleanup(); });
    
    m_descriptorCache.Init(m_device.logicalDevice);
    m_cleanupQueue.AddFunction([=] { m_descriptorCache.Cleanup(); });

    m_renderer.Init(&m_device, m_surface, m_windowExtent);
    m_cleanupQueue.AddFunction([=] { m_renderer.Cleanup(); });
    
    InitPipelines();
}

void EngineBase::InitSDL()
{
    SDL_Init(SDL_INIT_VIDEO);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);
	m_window = SDL_CreateWindow(
		"Vulkan Engine",            //window title
		SDL_WINDOWPOS_UNDEFINED,    // window position x (don't care)
		SDL_WINDOWPOS_UNDEFINED,    // window position y (don't care)
		m_windowExtent.width,       // window width in pixels
		m_windowExtent.height,      // window height in pixels
		window_flags 
	);

    // Cleanup
    m_cleanupQueue.AddFunction([=] { SDL_DestroyWindow(m_window); });
}

void EngineBase::InitVulkanCore() 
{
    // Vulkan instance
    vkb::InstanceBuilder builder;
    vkb::Instance vkbInst = builder
        .set_app_name("Vulkan Project")
        .require_api_version(1, 1, 0)
        .enable_validation_layers(true)
        .use_default_debug_messenger()
        .build()
        .value();
    m_instance = vkbInst.instance;
    m_debugMessenger = vkbInst.debug_messenger;

    // SDL Surface
    SDL_Vulkan_CreateSurface(m_window, m_instance, &m_surface);

    // Choose a GPU
    vkb::PhysicalDeviceSelector selector(vkbInst);
    vkb::PhysicalDevice vkbPhysDev = selector 
        .set_minimum_version(1, 1)
        .set_surface(m_surface)
        .select()
        .value();
    m_device.physicalDevice = vkbPhysDev.physical_device;

    // Vulkan logical device
    vkb::Device vkbDev = vkb::DeviceBuilder(vkbPhysDev)
        .build()
        .value();
    m_device.logicalDevice = vkbDev.device;
    
    // Device info
    m_device.features = vkbPhysDev.features;
    m_device.properties = vkbPhysDev.properties;
    printf("Using device %s\n", m_device.properties.deviceName);

    // Device queues
    m_device.queueFamilyProperties = vkbDev.queue_families;
    // Graphics queue
    auto graphRes = vkbDev.get_queue_index(vkb::QueueType::graphics);
    assert(graphRes.has_value());
    m_device.queueFamilies.graphics = graphRes.value();
    // Compute queue
    auto compRes = vkbDev.get_queue_index(vkb::QueueType::compute);
    if (compRes.has_value()) {
        m_device.queueFamilies.compute = compRes.value();
    }
    else {
        m_device.queueFamilies.compute = m_device.queueFamilies.graphics;   
    }
    // Transfer queue
    auto transRes = vkbDev.get_queue_index(vkb::QueueType::transfer);
    if (transRes.has_value()) {
        m_device.queueFamilies.transfer = transRes.value();
    }
    else {
        m_device.queueFamilies.transfer = m_device.queueFamilies.graphics;
    }
    // Check we chose the right queues
    assert(m_device.queueFamilyProperties[m_device.queueFamilies.graphics].queueFlags &
        VK_QUEUE_GRAPHICS_BIT);
    assert(m_device.queueFamilyProperties[m_device.queueFamilies.compute].queueFlags &
        VK_QUEUE_COMPUTE_BIT);
    assert(m_device.queueFamilyProperties[m_device.queueFamilies.transfer].queueFlags &
        VK_QUEUE_TRANSFER_BIT);

    // Cleanup
    m_cleanupQueue.AddFunction([=] {
        vkDestroyDevice(m_device.logicalDevice, nullptr); 
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr); 
        vkb::destroy_debug_utils_messenger(m_instance, m_debugMessenger);
        vkDestroyInstance(m_instance, nullptr);
    });
}

void EngineBase::InitVma() 
{
    VmaAllocatorCreateInfo info = {};
    info.instance = m_instance;
    info.physicalDevice = m_device.physicalDevice;
    info.device = m_device.logicalDevice;

    VK_CHECK(vmaCreateAllocator(&info, &m_vmaAllocator));
    m_cleanupQueue.AddFunction([=] { vmaDestroyAllocator(m_vmaAllocator); });
}

void EngineBase::InitPipelines() 
{    
    // Load the shaders
    vkw::Shader vertexShader, fragmentShader;
    vertexShader.Init(m_device.logicalDevice, "../shaders/triangle.vert.spv");
    fragmentShader.Init(m_device.logicalDevice, "../shaders/triangle.frag.spv");

    // allocate the camera buffer
    m_cameraBuffer.Init(m_vmaAllocator);
    m_cameraBuffer.Allocate(
        sizeof(GPUCameraData), 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    m_cleanupQueue.AddFunction([=] {
        m_cameraBuffer.Cleanup();
    });

    // Pipeline layout
    auto cameraBufferInfo = vkw::init::DescriptorBufferInfo(m_cameraBuffer.buffer, 0, sizeof(GPUCameraData));
    VkDescriptorSetLayout setLayout;
    vkw::DescriptorBuilder(&m_descriptorCache, &m_descriptorAllocator)
        .BindBuffer(0, &cameraBufferInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
        .Build(&m_firstSet, &setLayout);

    auto layoutInfo = vkw::init::PipelineLayoutCreateInfo();
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    VK_CHECK(vkCreatePipelineLayout(m_device.logicalDevice, &layoutInfo, nullptr, &m_pipelineLayout));
    
    // Create the pipeline
    vkw::GraphicsPipelineBuilder builder;
    builder.shaderStages.push_back(
        vkw::init::PipelineShaderStageCreateInfo(
            VK_SHADER_STAGE_VERTEX_BIT, vertexShader.shader));
    builder.shaderStages.push_back(
        vkw::init::PipelineShaderStageCreateInfo(
            VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader.shader));
    
    builder.vertexInputInfo = vkw::init::PipelineVertexInputStateCreateInfo();
    builder.inputAssembly = vkw::init::PipelineInputAssemblyStateCreateInfo(
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    
    builder.viewport.x = 0.0f;
    builder.viewport.y = 0.0f;
    builder.viewport.width = (float)m_windowExtent.width;
    builder.viewport.height = (float)m_windowExtent.height;
    builder.viewport.minDepth = 0.0f;
    builder.viewport.maxDepth = 1.0f;

    builder.scissors.offset = { 0, 0 };
    builder.scissors.extent = m_windowExtent;

    builder.rasterizer = vkw::init::PipelineRasterizationStateCreateInfo(
        VK_POLYGON_MODE_FILL);
    builder.multisampling = vkw::init::PipelineMultisampleStateCreateInfo();
    builder.colorAttachment = vkw::init::PipelineColorBlendAttachmentState(
        VK_COLOR_COMPONENT_R_BIT | 
        VK_COLOR_COMPONENT_G_BIT | 
        VK_COLOR_COMPONENT_B_BIT | 
        VK_COLOR_COMPONENT_A_BIT);

    builder.layout = m_pipelineLayout;
    builder.renderpass = m_renderer.swapchain.renderPass;

    VkResult res = builder.Build(m_device.logicalDevice, &m_pipeline);
    if (res != VK_SUCCESS) {
        printf("Error building graphics pipeline\n");
        assert(false);
    }

    // We can destroy the shaders right after creating the pipeline.
    vertexShader.Cleanup();
    fragmentShader.Cleanup();

    m_cleanupQueue.AddFunction([=] {
        vkDestroyPipeline(m_device.logicalDevice, m_pipeline, nullptr);
        vkDestroyPipelineLayout(m_device.logicalDevice, m_pipelineLayout, nullptr);
    });
}

void EngineBase::Run() 
{
    SDL_Event e;
    bool quit = false;

    while (!quit) {
        // poll SDL events
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }
        // Upload the camera data
        GPUCameraData* data = (GPUCameraData*)m_cameraBuffer.Map();
        data->color = { 0.0f, 1.0f, 1.0f, 0.0f };
        m_cameraBuffer.Unmap();

        // Draw a frame
        Renderer::DrawInfo drawInfo = {};
        drawInfo.pipeline = m_pipeline;
        drawInfo.pipelineLayout = m_pipelineLayout;
        drawInfo.descriptorSets = { m_firstSet };
        m_renderer.Draw(&drawInfo);
    }
}


void EngineBase::Cleanup() 
{
    // wait for all GPU operations to be over.
    VK_CHECK(vkDeviceWaitIdle(m_device.logicalDevice));

    // Destroy all vulkan objects/release memory.
    m_cleanupQueue.Flush();
}