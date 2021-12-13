#include "engine/engine.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include "VkBootstrap.h"
#include "engine/vk_check.h"


void Engine::Init() 
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
    m_cleanupQueue.AddFunction([=] { SDL_DestroyWindow(m_window); });

    InitVulkanCore();
    m_swapchain.Init(m_gpu, m_device, m_surface);
    m_cleanupQueue.AddFunction([=]{ m_swapchain.Cleanup(m_device); });
    m_frame.Init(m_device, m_graphicsQueueFamily);
    m_cleanupQueue.AddFunction([=]{ m_frame.Cleanup(m_device); });
    
    m_isInitialized = true;    
}

void Engine::InitVulkanCore() 
{
    // Vulkan instance
    vkb::InstanceBuilder builder;
    vkb::Instance vkbInst = builder
        .set_app_name("Vulkan Project")
        .request_validation_layers(true)
        .require_api_version(1, 1, 0)
        .use_default_debug_messenger()
        .build()
        .value();
    m_instance = vkbInst.instance;
    m_cleanupQueue.AddFunction([=] {
        vkDestroyInstance(m_instance, nullptr);
    });
    m_debugMessenger = vkbInst.debug_messenger;
    m_cleanupQueue.AddFunction([=] {
        vkb::destroy_debug_utils_messenger(m_instance, m_debugMessenger);	
    });

    // SDL Surface
    SDL_Vulkan_CreateSurface(m_window, m_instance, &m_surface);
    m_cleanupQueue.AddFunction([=] { 
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr); 
    });

    // Choose a GPU
    vkb::PhysicalDeviceSelector selector(vkbInst);
    vkb::PhysicalDevice vkbPhysDev = selector 
        .set_minimum_version(1, 1)
        .set_surface(m_surface)
        .select()
        .value();
    m_gpu = vkbPhysDev.physical_device;
    
    // Vulkan logical device
    vkb::Device vkbDev = vkb::DeviceBuilder(vkbPhysDev).build().value();
    m_device = vkbDev.device;
    m_cleanupQueue.AddFunction([=]{ 
        vkDestroyDevice(m_device, nullptr); 
    });

    // Graphics queue
    m_graphicsQueue = vkbDev.get_queue(vkb::QueueType::graphics).value();
	m_graphicsQueueFamily = vkbDev.get_queue_index(vkb::QueueType::graphics).value();
}

void Engine::Run() 
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
        // main engine logic is here
        Draw();
    }
}

void Engine::Draw() 
{
    
}

void Engine::Cleanup() 
{
    if (m_isInitialized) {
        m_cleanupQueue.Flush();
    }    
}