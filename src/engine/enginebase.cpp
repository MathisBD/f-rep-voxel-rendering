#include "engine/enginebase.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include "VkBootstrap.h"
#include "engine/vk_check.h"


void EngineBase::Init() 
{
    InitSDL();
    m_cleanupQueue.AddFunction([=] { SDL_DestroyWindow(m_window); });

    const vkb::Device& vkbDev = InitVulkanCore();
    m_cleanupQueue.AddFunction([=] {
        vkDestroyDevice(m_device, nullptr); 
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr); 
        vkb::destroy_debug_utils_messenger(m_instance, m_debugMessenger);
        vkDestroyInstance(m_instance, nullptr);
    });
    
    m_renderer.Init(vkbDev, m_windowExtent);
    m_cleanupQueue.AddFunction([=] {
        m_renderer.Cleanup(m_device);
    });

    m_isInitialized = true;    
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
    if (m_isInitialized) {
        m_cleanupQueue.Flush();
    }    
}