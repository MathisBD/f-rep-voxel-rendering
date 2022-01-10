#pragma once
#include "engine/voxel_storage.h"
#include "vk_wrapper/device.h"
#include "vk_wrapper/descriptor.h"
#include "third_party/vk_mem_alloc.h"
#include "utils/function_queue.h"
#include "vk_wrapper/buffer.h"




class Voxelizer
{
public:
    void Init(
        vkw::Device* device, 
        vkw::DescriptorAllocator* descAllocator, vkw::DescriptorLayoutCache* descCache,
        VoxelStorage* voxels, VmaAllocator vmaAllocator);
    void Cleanup();

    void Voxelize();
private:
    static const uint32_t THREAD_GROUP_SIZE_X    = 4;
    static const uint32_t THREAD_GROUP_SIZE_Y    = 4;
    static const uint32_t THREAD_GROUP_SIZE_Z    = 4;
    static const uint32_t MAX_LEVEL_COUNT        = 8;
    static const uint32_t MAX_CONSTANT_POOL_SIZE = 256;


    typedef struct {
        uint32_t dim;
        uint32_t nodeOfs;
        float cellSize;
        uint32_t _padding_;
    } ShaderLevelData;

    typedef struct {
        uint32_t levelCount;
        uint32_t level;    
        uint32_t tapeInstrCount;
        uint32_t _padding_;

        // The world positions of the grid bottom left corner.
        glm::vec3 gridWorldCoords;
        float gridWorldSize;
        
        ShaderLevelData levels[MAX_LEVEL_COUNT];
        float constantPool[MAX_CONSTANT_POOL_SIZE];
    } ShaderParams;

    vkw::Device* m_device;
    vkw::DescriptorAllocator* m_descAllocator;
    vkw::DescriptorLayoutCache* m_descCache;
    VmaAllocator m_vmaAllocator;
    VoxelStorage* m_voxels;
    FunctionQueue m_cleanupQueue;

    VkPipeline m_pipeline;
    VkPipelineLayout m_pipelineLayout;
    std::vector<VkDescriptorSet> m_descSets;

    VkQueue m_queue;
    VkCommandPool m_cmdPool;
    // The i-th semaphore is signaled when we are finished
    // voxelizing the i-th level.
    // It is waited upon before voxelizing the next level.
    std::vector<VkSemaphore> m_voxelizeLevelSems;
    // This fence is used to wait on every voxelize stage.
    VkFence m_fence;

    // The shader parameters uniform buffer.
    vkw::Buffer m_paramsBuffer;

    void InitCommands();
    void InitSynchronization();
    void InitBuffers();
    void InitPipelines();

    void AllocateGPUBuffers();
    void ZeroOutNodeBuffer();
    void UploadTape();

    void UpdateShaderParams(uint32_t level);
    void RecordCmd(VkCommandBuffer cmd, uint32_t level);
    void SubmitCmd(VkCommandBuffer cmd, uint32_t level);

    // This creates the level 'level',
    // assuming the previous level was created.
    // If level==0, then it only assumes an empty root node exists.
    void VoxelizeLevel(uint32_t level);
};