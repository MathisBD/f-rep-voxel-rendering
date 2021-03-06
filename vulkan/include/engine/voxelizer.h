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
        VoxelStorage* voxels);
    void Cleanup();

    void Voxelize(VkSemaphore waitSem, float tapeTime);
    // Returns a semaphore that is signaled when the voxelization is finished.
    VkSemaphore GetVoxelizeSem() const { return m_voxelizeLevelSems.back(); }    
    
    void PrintStats();
private:
    static const uint32_t THREAD_GROUP_SIZE      = 64;
    static const uint32_t MAX_LEVEL_COUNT        = 8;
    static const uint32_t MAX_CONSTANT_POOL_SIZE = 256;

    typedef struct {
        uint32_t dim;
        uint32_t nodeOfs;
        float cellSize;
        uint32_t _padding_;
    } ShaderLevelData;

    typedef struct {
        uint32_t level;    
        float tapeTime;
        uint32_t _padding_[2];

        // The world positions of the grid bottom left corner.
        glm::vec3 gridWorldCoords;
        float gridWorldSize;
        
        ShaderLevelData levels[MAX_LEVEL_COUNT];
        float constantPool[MAX_CONSTANT_POOL_SIZE];
    } ShaderParams;

    typedef struct {
        uint32_t childCount;
        uint32_t tapeIndex;
    } ShaderCounters;

    typedef struct {
        uint32_t tapeSizeSum[MAX_LEVEL_COUNT];
        uint32_t tapeSizeMax[MAX_LEVEL_COUNT];
    } ShaderStats;

    typedef struct {
        uint32_t maxSlotCount;
        uint32_t maxTapeSize;
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
    } ShaderVariant;

    vkw::Device* m_device;
    vkw::DescriptorAllocator* m_descAllocator;
    vkw::DescriptorLayoutCache* m_descCache;
    VoxelStorage* m_voxels;
    FunctionQueue m_cleanupQueue;

    std::vector<VkDescriptorSetLayout> m_descSetLayouts;
    std::vector<VkDescriptorSet> m_descSets;
    std::vector<ShaderVariant> m_shaders;

    VkQueue m_queue;
    VkCommandPool m_cmdPool;
    // The i-th semaphore is signaled when we are finished
    // voxelizing the i-th level.
    // It is waited upon before voxelizing the next level.
    std::vector<VkSemaphore> m_voxelizeLevelSems;
    // This fence is used to wait on every voxelize stage.
    VkFence m_fence;

    // Shader buffers.
    vkw::Buffer m_paramsBuffer;
    vkw::Buffer m_countersBuffer;
    vkw::Buffer m_statsBuffer;

    void InitCommands();
    void InitSynchronization();
    void InitBuffers();
    void InitDescSets();
    
    ShaderVariant CreateShaderVariant(uint32_t maxSlotCount, uint32_t maxTapeSize);
    ShaderVariant FindShaderVariant(uint32_t slotCount, uint32_t tapeSize);

    void AllocateGPUBuffers();
    void ZeroOutNodeBuffer();
    void UploadTape();

    void UpdateShaderParams(uint32_t level, float tapeTime);
    void UpdateShaderCounters(uint32_t level);
    void RecordCmd(VkCommandBuffer cmd, uint32_t level);
    void SubmitCmd(VkCommandBuffer cmd, uint32_t level, VkSemaphore waitSem);

    // This creates the level 'level',
    // assuming the previous level was created.
    // If level==0, then it only assumes an empty root node exists.
    void VoxelizeLevel(uint32_t level, VkSemaphore waitSem, float tapeTime);
};