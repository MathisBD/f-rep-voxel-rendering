#pragma once
#include "engine/voxel_storage.h"
#include <functional>


class SceneBuilder
{
public:
    void Init(VmaAllocator vmaAllocator, VoxelStorage* voxels,
        std::function<float(float, float, float)>&& density);
    void Cleanup();
    void BuildScene();
    void UploadSceneToGPU(VkCommandBuffer cmd);
private:
    class TreeNode 
    {
    public:
        uint32_t level;
        glm::u32vec3 coords;
        std::vector<uint32_t> mask;
        std::vector<uint32_t> maskPC;
        // Entries in childList are either :
        // -> pointers to TreeNodes (non-leaf node).
        // -> indices of Voxels in the voxel data vector (leaf node).
        std::vector<size_t> childList;
    };

    VmaAllocator m_allocator;
    VoxelStorage* m_voxels;
    std::function<float(float, float, float)> m_density;

    TreeNode* m_rootNode;
    std::vector<VoxelStorage::Voxel> m_voxelData;

    // These are CPU buffers, that will then get copied to the GPU buffers.
    struct {
        vkw::Buffer node;
        vkw::Buffer child;
        vkw::Buffer voxel;
    } m_stagingBuffers;

    // The memory mapped contents of the staging buffers.
    struct {
        void* node;
        void* child;
        void* voxel;
    } m_bufferContents;

    uint32_t Index3D(uint32_t x, uint32_t y, uint32_t z, uint32_t dim);

    // The coordinates are in the finest grid.
    size_t BuildVoxel(const glm::u32vec3& coords);
    void ComputeMaskPC(TreeNode* node);
    void CompactifyChildList(TreeNode* node);
    // The coordinates are in the finest grid AT THE NODE's level.
    void ComputeCoords(TreeNode* node, const glm::u32vec3& coords);
    // The coordinates are in the finest grid AT THE NODE's level.
    TreeNode* BuildNode(uint32_t level, const glm::u32vec3& coords);
    void DeleteNode(TreeNode* node);
    
    void BuildTrees();

    void CountNodes(TreeNode* node);

    void AllocateStagingBuffers();
    uint32_t LayoutNode(TreeNode* node, std::vector<uint32_t>& nextNodeIdx);    

    void AllocateGPUBuffers();    

    void PrintVoxelStats();
    void ForEachTreeNode(std::function<void(TreeNode*)> f);
    void ForEachTreeNodeHelper(TreeNode* node, std::function<void(TreeNode*)> f);
};