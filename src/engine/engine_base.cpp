#include "engine/engine_base.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include "VkBootstrap.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"
#include "engine/timer.h"



void EngineBase::Init() 
{
    InitSDL();
    InitVulkanCore();
    InitVma();
    InitImmUploadCtxt();

    m_inputManager.Init({ m_windowExtent.width, m_windowExtent.height });

    m_descriptorAllocator.Init(m_device.logicalDevice);
    m_cleanupQueue.AddFunction([=] { m_descriptorAllocator.Cleanup(); });
    
    m_descriptorCache.Init(m_device.logicalDevice);
    m_cleanupQueue.AddFunction([=] { m_descriptorCache.Cleanup(); });
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
        .add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT)
        .add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT)
        .add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT)
        //.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT)
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
    printf("[+] Using device %s\n", m_device.properties.deviceName);

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

void EngineBase::InitImmUploadCtxt() 
{
    uint32_t queueFamily = m_device.queueFamilies.graphics;

    // Queue
    vkGetDeviceQueue(m_device.logicalDevice, queueFamily, 0, &m_immUploadCtxt.queue);
    
    // Command pool
    auto poolInfo = vkw::init::CommandPoolCreateInfo(queueFamily);
    VK_CHECK(vkCreateCommandPool(m_device.logicalDevice, &poolInfo, nullptr, &m_immUploadCtxt.pool));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyCommandPool(m_device.logicalDevice, m_immUploadCtxt.pool, nullptr); 
    });

    // Fence
    auto fenceInfo = vkw::init::FenceCreateInfo();
    VK_CHECK(vkCreateFence(m_device.logicalDevice, &fenceInfo, nullptr, &m_immUploadCtxt.fence));
    m_cleanupQueue.AddFunction([=] { 
        vkDestroyFence(m_device.logicalDevice, m_immUploadCtxt.fence, nullptr);
    });
}

void EngineBase::Run() 
{
    SDL_Event e;
    bool quit = false;

    while (!quit) {
        Timer::UpdateTime();
        
        // poll SDL events
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
            else if (e.type == SDL_KEYUP || e.type == SDL_KEYDOWN) {
                m_inputManager.UpdateKey(e);
            }
        }
        m_inputManager.UpdateMouse();
        Draw();
    }
}


void EngineBase::Cleanup() 
{
    // wait for all GPU operations to be over.
    VK_CHECK(vkDeviceWaitIdle(m_device.logicalDevice));

    // Destroy all vulkan objects/release memory.
    m_cleanupQueue.Flush();
}

void EngineBase::ImmediateSubmit(
    std::function<void(VkCommandBuffer)>&& record) 
{
    VkCommandBuffer cmd;

    auto allocInfo = vkw::init::CommandBufferAllocateInfo(m_immUploadCtxt.pool);
    VK_CHECK(vkAllocateCommandBuffers(
        m_device.logicalDevice, &allocInfo, &cmd));

    auto beginInfo = vkw::init::CommandBufferBeginInfo(
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    record(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    auto submitInfo = vkw::init::SubmitInfo(&cmd);
    VK_CHECK(vkQueueSubmit(
        m_immUploadCtxt.queue, 1, &submitInfo, m_immUploadCtxt.fence));

    // Wait for the command to finish
    VK_CHECK(vkWaitForFences(
        m_device.logicalDevice, 1, &m_immUploadCtxt.fence, VK_TRUE, 1000000000));
    VK_CHECK(vkResetFences(
        m_device.logicalDevice, 1, &m_immUploadCtxt.fence));

    VK_CHECK(vkResetCommandPool(
        m_device.logicalDevice, m_immUploadCtxt.pool, 0));
}