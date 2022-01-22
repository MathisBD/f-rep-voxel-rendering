#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "vk_wrapper/buffer.h"
#include "vk_wrapper/device.h"
#include "engine/render_target.h"
#include "engine/camera.h"
#include "engine/voxel_storage.h"
#include "utils/function_queue.h"
#include "vk_wrapper/descriptor.h"




class Raytracer
{
public:
    void Init(
        vkw::Device* device, 
        vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
        RenderTarget* target, VoxelStorage* voxels);
    void Cleanup();

    void Trace(VkSemaphore waitSem, const Camera* camera, float tapeTime);
    // Returns a semaphore that is signaled 
    // when the raytracing is finished.
    VkSemaphore GetTraceSem() const { return m_semaphore; }
    void SetBackgroundColor(const glm::vec3& color);
private:
    static const uint32_t THREAD_GROUP_SIZE_X = 16;
    static const uint32_t THREAD_GROUP_SIZE_Y = 16;
    
    // A single point light.
    typedef struct {
        glm::vec4 color;
        // The direction the light is pointing (w unused).
        glm::vec4 direction;
    } ShaderLight;

    typedef struct {
        uint32_t dim;
        uint32_t nodeOfs;
        float cellSize;
        uint32_t _padding_;
    } ShaderLevelData;

    typedef struct {
        float time;
        float tapeTime;
        uint32_t outImgLayer;
        uint32_t _padding_;

        // The camera world position (w unused).
        glm::vec4 cameraPosition;
        // The direction the camera is looking in (w unused).
        // The camera forward, up and right vectors are normalized
        // and orthogonal.
        glm::vec4 cameraForward;
        glm::vec4 cameraUp;
        glm::vec4 cameraRight;
        
        // The world positions of the grid bottom left corner.
        glm::vec3 gridWorldCoords;
        float gridWorldSize;

        // The screen resolution in pixels.
        glm::uvec2 screenResolution;
        // The world size of the screen boundaries
        // at one unit away from the camera position.
        glm::vec2 screenWorldSize;  
        glm::vec4 backgroundColor;
    
        ShaderLevelData levels[8];
        ShaderLight lights[8];
        float constPool[256];
    } ShaderParams;

    
    FunctionQueue m_cleanupQueue;
    vkw::Device* m_device;
    RenderTarget* m_target;
    VoxelStorage* m_voxels;
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
    vkw::Buffer m_paramsBuffer;
    uint32_t m_targetImgLayer = 0;

    void InitCommands();
    void InitSynchronization();
    void InitPipeline();
    void InitBuffers();
    void InitUploadCtxt();

    void UpdateShaderParams(const Camera* camera, float tapeTime);

    void RecordComputeCmd(VkCommandBuffer cmd);
    void SubmitComputeCmd(VkCommandBuffer cmd, VkSemaphore renderSem);
    void ImmediateSubmit(std::function<void(VkCommandBuffer)>&& record);
};