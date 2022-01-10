#pragma once
#include "engine/voxel_storage.h"
#include "engine/csg_expression.h"
#include "engine/csg_tape.h"
#include "utils/thread_pool.h"


class SceneBuilder
{
public:
    void Init(VmaAllocator vmaAllocator, VoxelStorage* voxels);
    void Cleanup();
    void BuildScene();
    void UploadSceneToGPU(VkCommandBuffer cmd);
private:
    class TreeNode 
    {
    public:
        uint32_t level;
        std::vector<uint32_t> leafMask;
        std::vector<uint32_t> interiorMask;
        // The child list only contains the interior node children,
        // not the leaf node children.
        std::vector<TreeNode*> childList;
    };

    VmaAllocator m_allocator;
    VoxelStorage* m_voxels;

    ThreadPool m_threadPool = { std::thread::hardware_concurrency() };
    TreeNode* m_rootNode;

    // These are CPU buffers, that will then get copied to the GPU buffers.
    struct {
        vkw::Buffer node;
        vkw::Buffer tape;
    } m_stagingBuffers;

    // The memory mapped contents of the staging buffers.
    struct {
        void* node;
        void* tape;
    } m_bufferContents;

    uint32_t Index3D(uint32_t x, uint32_t y, uint32_t z, uint32_t dim);

    bool HasAllLeafChildren(TreeNode* node);
    // The coordinates are in the finest grid AT THE NODE's level.
    TreeNode* BuildNode(uint32_t level, const glm::u32vec3& coords);
    void DeleteNode(TreeNode* node);

    void CountInteriorNodes(TreeNode* node);

    void AllocateStagingBuffers();
    uint32_t LayoutNode(TreeNode* node, std::vector<uint32_t>& nextNodeIdx);    

    void AllocateGPUBuffers();    

    void PrintVoxelStats();
    void ForEachTreeNode(std::function<void(TreeNode*)> f);
    void ForEachTreeNodeHelper(TreeNode* node, std::function<void(TreeNode*)> f);
};