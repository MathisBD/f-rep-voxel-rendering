#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "vk_wrapper/buffer.h"
#include "vk_wrapper/device.h"
#include "engine/render_target.h"
#include "engine/camera.h"
#include "engine/cube_grid.h"
#include "engine/cleanup_queue.h"
#include "vk_wrapper/descriptor.h"



class Raytracer
{
public:
    void Init(
        vkw::Device* device, RenderTarget* target, VmaAllocator vmaAllocator,
        vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache);
    void Cleanup();

private:
    CleanupQueue m_cleanupQueue;
    vkw::Device* m_device;
    RenderTarget* m_target;
    VmaAllocator m_vmaAllocator;
    vkw::DescriptorAllocator* m_descAllocator;
    vkw::DescriptorLayoutCache* m_descCache;

    VkPipeline m_pipeline;
    VkPipelineLayout m_pipelineLayout;
    std::vector<VkDescriptorSet> m_descSets;
    
    VkQueue m_queue;
    VkCommandPool m_cmdPool;
    // signaled when the compute command is finished
    VkSemaphore m_semaphore;
    // signaled when the compute command is finished
    VkFence m_fence; 

    vkw::Buffer m_ddaUniforms;
    vkw::Buffer m_ddaVoxels;
    CubeGrid m_voxelGrid;
    vkw::Buffer m_ddaLights;
    const size_t m_lightCount = 2;

    Camera m_camera;

    void InitCommands();
    void InitSynchronization();
    void InitPipeline();
    void InitBuffers();


};