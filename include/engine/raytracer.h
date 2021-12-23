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
        vkw::Device* device, 
        vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
        RenderTarget* target, VmaAllocator vmaAllocator);
    void Cleanup();

    void Trace(VkSemaphore renderSem, const Camera* camera);
    VkSemaphore GetComputeSemaphore() const { return m_semaphore; }
    void SetBackgroundColor(const glm::vec3& color);
private:
    typedef struct {
        glm::vec4 color;
        // The normal vector at the center of the voxel (w unused).
        glm::vec4 normal;
    } DDAVoxel;

    // A single point light.
    typedef struct {
        glm::vec4 color;
        // The direction the light is pointing (w unused).
        glm::vec4 direction;
    } DDALight;

    typedef struct {     
        // The screen resolution in pixels (zw unused).
        glm::uvec4 screenResolution;
        // The world size of the screen boundaries
        // at one unit away from the camera position (zw unused).
        glm::vec4 screenWorldSize;

        // The camera world position (w unused).
        glm::vec4 cameraPosition;
        // The direction the camera is looking in (w unused).
        // The camera forward, up and right vectors are normalized
        // and orthogonal.
        glm::vec4 cameraForward;
        glm::vec4 cameraUp;
        glm::vec4 cameraRight;

        // The world positions of the grid bottom left corner (xyz)
        // and the world size of the grid (w).
        glm::vec4 gridWorldCoords;
        // The number of subdivisions along each grid axis (yzw unused).
        glm::uvec4 gridResolution;

        // The color we use for rays that don't intersect any voxel (w unused).
        glm::vec4 backgroundColor;
        glm::uvec4 lightCount;
        DDALight lights[8];
    } DDAUniforms;

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

    // The context used for submitting an immediate command.
    struct {
        VkCommandPool cmdPool;
        VkFence fence;
    } m_uploadCtxt;

    glm::vec3 m_backgroundColor = { 0.0f, 0.0f, 0.0f };
    size_t m_lightCount = 0;
    CubeGrid m_voxelGrid;

    struct {
        vkw::Buffer uniforms;
        vkw::Buffer voxelData;
        vkw::Buffer voxelMask;
        vkw::Buffer voxelMaskPC;
    } m_buffers;

    void InitCommands();
    void InitSynchronization();
    void InitPipeline();
    void InitBuffers();
    void InitUploadCtxt();

    void UpdateUniformBuffer(const Camera* camera);
    void UpdateVoxelBuffers(std::function<float(float, float, float)>&& density);

    void RecordComputeCmd(VkCommandBuffer cmd);
    void SubmitComputeCmd(VkCommandBuffer cmd, VkSemaphore renderSem);
    void ImmediateSubmit(std::function<void(VkCommandBuffer)>&& record);
};