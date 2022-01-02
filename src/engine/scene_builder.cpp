#include "engine/scene_builder.h"
#include <string.h>


void SceneBuilder::Init(VmaAllocator vmaAllocator, VoxelStorage* voxels,
    std::function<float(float, float, float)>&& density) 
{
    m_allocator = vmaAllocator;
    m_voxels = voxels;
    m_density = density;

    m_stagingBuffers.node.Init(m_allocator);
    m_stagingBuffers.child.Init(m_allocator);
    m_stagingBuffers.voxel.Init(m_allocator);
}

void SceneBuilder::Cleanup() 
{
    m_stagingBuffers.node.Cleanup();
    m_stagingBuffers.child.Cleanup();
    m_stagingBuffers.voxel.Cleanup();
}

void SceneBuilder::BuildScene() 
{
    m_rootNode = BuildNode(0, { 0, 0, 0 });
    CountNodes(m_rootNode);
    AllocateStagingBuffers();

    m_bufferContents.node = m_stagingBuffers.node.Map(); 
    m_bufferContents.child = m_stagingBuffers.child.Map(); 
    m_bufferContents.voxel = m_stagingBuffers.voxel.Map(); 

    std::vector<uint32_t> nextNodeIdx(m_voxels->gridLevels, 0);
    LayoutNode(m_rootNode, nextNodeIdx);
    // layout the voxels
    memcpy(m_bufferContents.voxel, m_voxelData.data(), 
        m_voxelData.size() * sizeof(VoxelStorage::Voxel));

    m_stagingBuffers.node.Unmap();
    m_stagingBuffers.child.Unmap();
    m_stagingBuffers.voxel.Unmap();
}

void SceneBuilder::UploadSceneToGPU(VkCommandBuffer cmd) 
{
    // Don't forget to allocate on the GPU.
    AllocateGPUBuffers();
    
    VkBufferCopy nodeRegion;
    nodeRegion.srcOffset = 0;
    nodeRegion.dstOffset = 0;
    nodeRegion.size = m_stagingBuffers.node.size;
    vkCmdCopyBuffer(cmd, m_stagingBuffers.node.buffer, m_voxels->nodeBuffer.buffer, 1, &nodeRegion);

    VkBufferCopy childRegion;
    childRegion.srcOffset = 0;
    childRegion.dstOffset = 0;
    childRegion.size = m_stagingBuffers.child.size;
    vkCmdCopyBuffer(cmd, m_stagingBuffers.child.buffer, m_voxels->childBuffer.buffer, 1, &childRegion);

    VkBufferCopy voxelRegion;
    voxelRegion.srcOffset = 0;
    voxelRegion.dstOffset = 0;
    voxelRegion.size = m_stagingBuffers.voxel.size;
    vkCmdCopyBuffer(cmd, m_stagingBuffers.voxel.buffer, m_voxels->voxelBuffer.buffer, 1, &voxelRegion); 
}


void SceneBuilder::ComputeMaskPC(TreeNode* node) 
{
    uint32_t dim = m_voxels->gridDims[node->level];

    uint32_t partialCount = 0;
    for (uint32_t q = 0; q < ((dim*dim*dim) / 32); q++) {
        partialCount += __builtin_popcount(node->mask[q]);
        node->maskPC[q] = partialCount;
    }
}

void SceneBuilder::CompactifyChildList(TreeNode* node) 
{
    uint32_t dim = m_voxels->gridDims[node->level];

    uint32_t pos = 0;    
    for (uint32_t index = 0; index < dim*dim*dim; index++) {
        uint32_t q = index >> 5;
        uint32_t r = index & ((1 << 5) - 1);
        if (node->mask[q] & (1 << r)) {
            node->childList[pos] = node->childList[index];
            pos++;
        }
    }
}


// Returns the index of the voxel in the voxel data vector.
// Returns the maximum size_t number if no voxel was built.
size_t SceneBuilder::BuildVoxel(const glm::u32vec3& coords) 
{
    glm::vec3 pos = m_voxels->WorldPosition(coords);
    float d = m_density(pos.x, pos.y, pos.z);

    if (d < 0.0f) {
        return std::numeric_limits<size_t>::max();
    }

    VoxelStorage::Voxel voxel;
    float eps = 0.0001f;
    voxel.normal = -glm::normalize(glm::vec3(
        (m_density(pos.x + eps, pos.y, pos.z) - d) / eps,
        (m_density(pos.x, pos.y + eps, pos.z) - d) / eps,
        (m_density(pos.x, pos.y, pos.z + eps) - d) / eps ));
    voxel.materialIndex = 0;    

    // The real voxel will live in the vector.
    m_voxelData.push_back(std::move(voxel));
    
    return m_voxelData.size() - 1;
}

void SceneBuilder::ComputeCoords(TreeNode* node, const glm::u32vec3& coords) 
{
    // Compute the total dimension of the child nodes.
    uint32_t dim = 1;
    for (uint32_t level = node->level; level < m_voxels->gridLevels; level++) 
    {
        dim *= m_voxels->gridDims[level];
    }
    node->coords = coords * dim;    
}

SceneBuilder::TreeNode* SceneBuilder::BuildNode(uint32_t level, const glm::u32vec3& coords) 
{
    assert(level < m_voxels->gridLevels);
    uint32_t dim = m_voxels->gridDims[level];

    // Allocate the node
    TreeNode* node = new TreeNode();
    node->level = level;
    node->mask = std::vector<uint32_t>((dim*dim*dim) / 32);
    node->maskPC = std::vector<uint32_t>((dim*dim*dim) / 32);
    node->childList = std::vector<size_t>(dim*dim*dim);

    // Recurse on the children / create the voxels
    for (uint32_t x = 0; x < dim; x++) {
        for (uint32_t y = 0; y < dim; y++) {
            for (uint32_t z = 0; z < dim; z++) {
                uint32_t index = Index3D(x, y, z, dim);
                glm::u32vec3 childCoords = { x, y, z };
                if (level > 0) {
                    childCoords += coords * m_voxels->gridDims[level-1];    
                }

                bool hasChild;
                // Voxel.
                if (level == m_voxels->gridLevels - 1) {
                    size_t voxelId = BuildVoxel(childCoords);
                    hasChild = (voxelId != std::numeric_limits<size_t>::max());
                    node->childList[index] = voxelId;
                }
                // Child.
                else {
                    TreeNode* childPtr = BuildNode(level + 1, childCoords); 
                    hasChild = (childPtr != nullptr); 
                    node->childList[index] = (size_t)childPtr;
                }
                // Update the node mask
                if (hasChild) {
                    uint32_t q = index >> 5;
                    uint32_t r = index & ((1 << 5) - 1);
                    node->mask[q] |= 1 << r;
                }
            }
        }
    }

    ComputeMaskPC(node);
    // The node is empty
    if (node->maskPC.back() == 0) {
        free(node);
        return nullptr;
    }
    CompactifyChildList(node);
    ComputeCoords(node, coords);
    return node;
}

uint32_t SceneBuilder::Index3D(uint32_t x, uint32_t y, uint32_t z, uint32_t dim) 
{
    return z + y * dim + x * dim * dim;    
}

void SceneBuilder::CountNodes(TreeNode* node) 
{
    m_voxels->nodeCount[node->level]++;
 
    if (node->level < m_voxels->gridLevels - 1) {   
        uint32_t childCount = node->maskPC.back();
        for (uint32_t pos = 0; pos < childCount; pos++) {
            CountNodes((TreeNode*)(node->childList[pos]));
        }
    }
}

void SceneBuilder::AllocateStagingBuffers() 
{
    uint32_t voxelSize = m_voxelData.size() * sizeof(VoxelStorage::Voxel);
    uint32_t nodeSize = 0;
    uint32_t childSize = 0;
    for (uint32_t level = 0; level < m_voxels->gridLevels; level++) {
        uint32_t dim = m_voxels->gridDims[level];
        nodeSize += m_voxels->nodeCount[level] * m_voxels->NodeSize(level);
        childSize += m_voxels->nodeCount[level] * (dim * dim * dim) * sizeof(uint32_t);
    }

    m_stagingBuffers.node.Allocate(nodeSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_stagingBuffers.child.Allocate(childSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_stagingBuffers.voxel.Allocate(voxelSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);    
}


void SceneBuilder::AllocateGPUBuffers() 
{
    // We assume the staging buffers were already allocated.
    VkBufferUsageFlags bufferUsage = 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    m_voxels->nodeBuffer.Allocate(m_stagingBuffers.node.size, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    m_voxels->childBuffer.Allocate(m_stagingBuffers.child.size, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    m_voxels->voxelBuffer.Allocate(m_stagingBuffers.voxel.size, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);    
}

uint32_t SceneBuilder::LayoutNode(TreeNode* node, std::vector<uint32_t>& nextNodeIdx) 
{
    uint32_t dim = m_voxels->gridDims[node->level];

    // claim a node index
    uint32_t idx = nextNodeIdx[node->level]++;
    assert(idx < m_voxels->nodeCount[node->level]);

    // Compute the address of the node in the staging buffer.
    size_t nodeAddr = (size_t)m_bufferContents.node;
    for (uint32_t i = 0; i < node->level; i++) {
        nodeAddr += m_voxels->nodeCount[i] * m_voxels->NodeSize(i);
    }
    nodeAddr += idx * m_voxels->NodeSize(node->level);

    // Set the node's child list index.
    // This is for now the same as the node index.
    uint32_t* clIdx = (uint32_t*)(nodeAddr + m_voxels->NodeOfsCLIdx(node->level));
    *clIdx = idx;
    // Copy the coordinates.
    uint32_t* coords = (uint32_t*)(nodeAddr + m_voxels->NodeOfsCoords(node->level));
    coords[0] = node->coords.x;
    coords[1] = node->coords.y;
    coords[2] = node->coords.z;
    // Copy the mask
    memcpy((void*)(nodeAddr + m_voxels->NodeOfsMask(node->level)), 
        node->mask.data(), 
        node->mask.size() * sizeof(uint32_t));
    // Copy the mask PC
    memcpy((void*)(nodeAddr + m_voxels->NodeOfsMaskPC(node->level)),
        node->maskPC.data(),
        node->maskPC.size() * sizeof(uint32_t));

    // Compute the address of the node's child list in the child buffer.
    size_t childAddr = (size_t)m_bufferContents.child;
    for (uint32_t i = 0; i < node->level; i++) {
        childAddr += m_voxels->nodeCount[i] * (dim * dim * dim) * sizeof(uint32_t);
    }
    childAddr += idx * (dim * dim * dim) * sizeof(uint32_t);
    uint32_t* childList = (uint32_t*)childAddr;

    
    // Recurse on children
    uint32_t childCount = node->maskPC.back();
    for (uint32_t pos = 0; pos < childCount; pos++) {
        uint32_t childIdx;
        if (node->level < m_voxels->gridLevels - 1) {
            TreeNode* childNode = (TreeNode*)node->childList[pos];
            childIdx = LayoutNode(childNode, nextNodeIdx);
        }
        else {
            childIdx = (uint32_t)node->childList[pos];
        }
        childList[pos] = childIdx;
    }


    // Print the node.
    {
        //uint32_t dim = m_voxels->gridDims[node->level];

        printf("%s:%u level=%d\tcoords=%u %u %u\n", 
            node->level == m_voxels->gridLevels - 1 ? "LEAF" : "INTERIOR",
            idx, node->level, coords[0], coords[1], coords[2]);
        printf("coords offset (in bytes)=%lu\n", (size_t)coords - (size_t)m_bufferContents.node);
        /*printf("\tmask = ");
        assert(node->mask.size() == (dim*dim*dim) / 32);
        for (size_t i = 0; i < (dim*dim*dim) / 32; i++) {
            printf("%x ", node->mask[i]);
        }
        printf("\n");

        printf("\tmask PC = ");
        assert(node->maskPC.size() == (dim*dim*dim) / 32);
        for (size_t i = 0; i < (dim*dim*dim) / 32; i++) {
            printf("%u ", node->maskPC[i]);
        }
        printf("\n");*/

        printf("\tchildren = ");
        for (size_t i = 0; i < node->maskPC.back(); i++) {
            printf("%u ", childList[i]);
        }
        printf("\n\n");
    }

    return idx;
}
