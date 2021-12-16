#include "engine/enginebase.h"
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
    m_cleanupQueue.AddFunction([=] { SDL_DestroyWindow(m_window); });

    InitVulkanCore();
    m_cleanupQueue.AddFunction([=] {
        vkDestroyDevice(m_device.logicalDevice, nullptr); 
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr); 
        vkb::destroy_debug_utils_messenger(m_instance, m_debugMessenger);
        vkDestroyInstance(m_instance, nullptr);
    });
    
    m_renderer.Init(vkbDev, m_windowExtent);
    m_cleanupQueue.AddFunction([=] {
        m_renderer.Cleanup(m_device);
    });

    InitPipelines();
    m_cleanupQueue.AddFunction([=] {
        vkDestroyPipeline(m_device.logicalDevice, m_pipeline, nullptr);
    });
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
}

void EngineBase::InitVulkanCore() 
{
    // Vulkan instance
    vkb::InstanceBuilder builder;
    vkb::Instance vkbInst = builder
        .set_app_name("Vulkan Project")
        .require_api_version(1, 1, 0)
        .enable_validation_layers(true)
        .add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT)
        .add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT)
        .add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_MAX_ENUM_EXT)
        .add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT)
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

    // Device queues
    m_device.queueFamilyProperties = vkbDev.queue_families;
    // Graphics queue : default for everything
    m_device.queueFamilies.graphics = vkbDev.get_queue_index(vkb::QueueType::graphics);
    m_device.queueFamilies.compute = m_device.queueFamilies.graphics;
    m_device.queueFamilies.transfer = m_device.queueFamilies.graphics;
    // Compute queue
    auto compRes = vkbDev.get_queue_index(vkb::QueueType::compute);
    if (compRes.has_value()) {
        m_device.queueFamilies.compute = compRes.value();
    }
    // Transfer queue
    auto transRes = vkbDev.get_queue_index(vkb::QueueType::transfer);
    if (transRes.has_value()) {
        m_device.queueFamilies.transfer = transRes.value();
    }
}

void EngineBase::InitPipelines() 
{
    // Load the shaders
    vkw::Shader vertexShader;
    vkw::Shader fragmentShader;
    vertexShader.Init(m_device.logicalDevice, "../shaders/triangle.vert");
    fragmentShader.Init(m_device.logicalDevice, "../shaders/triangle.frag");

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

    // empty layout (no descriptor sets or push constants)
    auto layoutInfo = vkw::init::PipelineLayoutCreateInfo();
    VK_CHECK(vkCreatePipelineLayout(m_device.logicalDevice, &layoutInfo, nullptr, &builder.layout));

    builder.renderpass = m_renderer.swapchain.renderPass;

    VkResult res = builder.Build(m_device.logicalDevice, &m_pipeline);
    if (res != VK_SUCCESS) {
        printf("Error building graphics pipeline\n");
        assert(false);
    }
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
        // Draw a frame
        m_renderer.Draw(m_device);
    }
}


void EngineBase::Cleanup() 
{
    m_cleanupQueue.Flush();
}