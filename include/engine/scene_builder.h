#pragma once
#include "engine/voxel_storage.h"
#include "engine/csg_expression.h"
#include "engine/csg_tape.h"


class SceneBuilder
{
public:
    void Init(VmaAllocator vmaAllocator, VoxelStorage* voxels,
        csg::Expr shape);
    void Cleanup();
    void BuildScene();
    void UploadSceneToGPU(VkCommandBuffer cmd);
private:
    class TreeNode 
    {
    public:
        uint32_t level;
        glm::u32vec3 coords;
        std::vector<uint32_t> interiorMask;
        std::vector<uint32_t> interiorMaskPC;
        std::vector<uint32_t> leafMask;
        // The child list only contains the interior node children,
        // no the leaf node children.
        std::vector<TreeNode*> childList;
    };

    VmaAllocator m_allocator;
    VoxelStorage* m_voxels;
    csg::Expr m_shape;

    TreeNode* m_rootNode;

    // These are CPU buffers, that will then get copied to the GPU buffers.
    struct {
        vkw::Buffer node;
        vkw::Buffer child;
        vkw::Buffer tape;
    } m_stagingBuffers;

    // The memory mapped contents of the staging buffers.
    struct {
        void* node;
        void* child;
        void* tape;
    } m_bufferContents;

    uint32_t Index3D(uint32_t x, uint32_t y, uint32_t z, uint32_t dim);

    void ComputeInteriorMaskPC(TreeNode* node);
    void CompactifyChildList(TreeNode* node);
    // The coordinates are in the finest grid AT THE NODE's level.
    void ComputeCoords(TreeNode* node, const glm::u32vec3& coords);
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