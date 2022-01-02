#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "vk_wrapper/buffer.h"
#include "vk_wrapper/device.h"
#include "engine/render_target.h"
#include "engine/camera.h"
#include "engine/voxel_storage.h"
#include "engine/cleanup_queue.h"
#include "vk_wrapper/descriptor.h"



#define MAX_LIGHT_COUNT     8
#define MAX_MATERIAL_COUNT  8
#define MAX_LEVEL_COUNT     8

typedef struct {
    // The normal vector at the center of the voxel (w unused).
    glm::vec3 normal;
    uint32_t materialIndex;
} ShaderVoxel;

// A single point light.
typedef struct {
    glm::vec4 color;
    // The direction the light is pointing (w unused).
    glm::vec4 direction;
} ShaderLight;

typedef struct {
    uint32_t dim;
    uint32_t nodeOfs;
    uint32_t childOfs;
    float cellSize;
} ShaderLevelData;

typedef struct {
    glm::vec4 color;
} ShaderMaterial;

typedef struct {
    uint32_t lightCount;
    uint32_t materialCount;
    uint32_t levelCount;  
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

    ShaderLevelData levels[MAX_LEVEL_COUNT];

    // The color we use for rays that don't intersect any voxel (w unused).
    glm::vec4 backgroundColor;
    ShaderLight lights[MAX_LIGHT_COUNT];
    ShaderMaterial materials[MAX_MATERIAL_COUNT];
} ShaderParams;


class Raytracer
{
public:
    void Init(
        vkw::Device* device, 
        vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
        RenderTarget* target, VoxelStorage* voxels, VmaAllocator vmaAllocator);
    void Cleanup();

    void Trace(VkSemaphore renderSem, const Camera* camera);
    VkSemaphore GetComputeSemaphore() const { return m_semaphore; }
    void SetBackgroundColor(const glm::vec3& color);
private:
    CleanupQueue m_cleanupQueue;
    vkw::Device* m_device;
    RenderTarget* m_target;
    VoxelStorage* m_voxels;
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
    vkw::Buffer m_paramsBuffer;

    void InitCommands();
    void InitSynchronization();
    void InitPipeline();
    void InitBuffers();
    void InitUploadCtxt();

    void UpdateShaderParams(const Camera* camera);

    void RecordComputeCmd(VkCommandBuffer cmd);
    void SubmitComputeCmd(VkCommandBuffer cmd, VkSemaphore renderSem);
    void ImmediateSubmit(std::function<void(VkCommandBuffer)>&& record);

    uint32_t SplitBy3(uint32_t x);
    uint32_t MortonEncode(glm::u32vec3 cell);
};