#include "engine/scene_builder.h"
#include <string.h>


void SceneBuilder::Init(VmaAllocator vmaAllocator, VoxelStorage* voxels,
    csg::Expr shape) 
{
    m_allocator = vmaAllocator;
    m_voxels = voxels;
    m_shape = shape;
    m_voxels->tape = csg::Tape(shape);

    m_stagingBuffers.node.Init(m_allocator);
    m_stagingBuffers.child.Init(m_allocator);
    m_stagingBuffers.tape.Init(m_allocator);
    m_stagingBuffers.constants.Init(m_allocator);
}

void SceneBuilder::Cleanup() 
{
    m_stagingBuffers.node.Cleanup();
    m_stagingBuffers.child.Cleanup();
    m_stagingBuffers.tape.Cleanup();
    m_stagingBuffers.constants.Cleanup();
}

void SceneBuilder::BuildScene() 
{
    m_rootNode = BuildNode(0, { 0, 0, 0 });
    // The scene shouldn't be empty.
    assert(m_rootNode);

    CountInteriorNodes(m_rootNode);
    AllocateStagingBuffers();

    m_bufferContents.node = m_stagingBuffers.node.Map(); 
    m_bufferContents.child = m_stagingBuffers.child.Map(); 
    m_bufferContents.tape = m_stagingBuffers.tape.Map();
    m_bufferContents.constants = m_stagingBuffers.constants.Map();

    std::vector<uint32_t> nextNodeIdx(m_voxels->gridLevels, 0);
    LayoutNode(m_rootNode, nextNodeIdx);
    // layout the tape
    memcpy(m_bufferContents.tape, m_voxels->tape.instructions.data(), 
        m_voxels->tape.instructions.size() * sizeof(csg::Tape::Instr));
    // layout the tape constants
    memcpy(m_bufferContents.constants, m_voxels->tape.constantPool.data(),
        m_voxels->tape.constantPool.size() * sizeof(float));

    m_stagingBuffers.node.Unmap();
    m_stagingBuffers.child.Unmap();
    m_stagingBuffers.tape.Unmap();
    m_stagingBuffers.constants.Unmap();

    // Print some stats before we cleanup.
    PrintVoxelStats();

    // We can already delete the nodes.
    DeleteNode(m_rootNode);
    m_rootNode = nullptr;
}

void SceneBuilder::DeleteNode(TreeNode* node) 
{
    if (node->level < m_voxels->gridLevels - 1) {
        uint32_t childCount = node->interiorMaskPC.back();
        for (uint32_t i = 0; i < childCount; i++) {
            DeleteNode(node->childList[i]);
        }
    }
    delete node;    
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

    VkBufferCopy tapeRegion;
    tapeRegion.srcOffset = 0;
    tapeRegion.dstOffset = 0;
    tapeRegion.size = m_stagingBuffers.tape.size;
    vkCmdCopyBuffer(cmd, m_stagingBuffers.tape.buffer, m_voxels->tapeBuffer.buffer, 1, &tapeRegion);
    
    VkBufferCopy constantsRegion;
    constantsRegion.srcOffset = 0;
    constantsRegion.dstOffset = 0;
    constantsRegion.size = m_stagingBuffers.constants.size;
    vkCmdCopyBuffer(cmd, m_stagingBuffers.constants.buffer, m_voxels->constPoolBuffer.buffer, 1, &constantsRegion);
}


void SceneBuilder::ComputeInteriorMaskPC(TreeNode* node) 
{
    uint32_t dim = m_voxels->gridDims[node->level];

    uint32_t partialCount = 0;
    for (uint32_t q = 0; q < ((dim*dim*dim) / 32); q++) {
        partialCount += __builtin_popcount(node->interiorMask[q]);
        node->interiorMaskPC[q] = partialCount;
    }
}

void SceneBuilder::CompactifyChildList(TreeNode* node) 
{
    uint32_t dim = m_voxels->gridDims[node->level];

    uint32_t pos = 0;    
    for (uint32_t index = 0; index < dim*dim*dim; index++) {
        uint32_t q = index >> 5;
        uint32_t r = index & ((1 << 5) - 1);
        if (node->interiorMask[q] & (1 << r)) {
            node->childList[pos] = node->childList[index];
            pos++;
        }
    }
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
    // Check there is at least one uint in the mask.
    // We could treat this as a special case but it would make
    // calculating the node size/offsets more complex for no real benefit.
    assert((dim*dim*dim) / 32 > 0);
    node->leafMask = std::vector<uint32_t>((dim*dim*dim) / 32, 0);
    if (node->level < m_voxels->gridLevels - 1) {
        node->interiorMask = std::vector<uint32_t>((dim*dim*dim) / 32, 0);
        node->interiorMaskPC = std::vector<uint32_t>((dim*dim*dim) / 32, 0);
        node->childList = std::vector<TreeNode*>(dim*dim*dim, nullptr);
    }

    // Recurse on the children / create the voxels
    bool hasChild = false;
    for (uint32_t x = 0; x < dim; x++) {
        for (uint32_t y = 0; y < dim; y++) {
            for (uint32_t z = 0; z < dim; z++) {
                uint32_t index = Index3D(x, y, z, dim);
                glm::u32vec3 childCoords = { x, y, z };
                childCoords += coords * m_voxels->gridDims[level];

                uint32_t q = index >> 5;
                uint32_t r = index & ((1 << 5) - 1);

                // Max level : check if the child is a leaf node.
                if (level == m_voxels->gridLevels - 1) {
                    glm::vec3 pos = m_voxels->WorldPosition(childCoords);
                    float density = m_voxels->tape.Eval(pos.x, pos.y, pos.z);
                    // Simply update the leaf mask
                    if (density < 0.0f) {
                        node->leafMask[q] |= (1 << r);
                        hasChild = true;
                    }
                }
                // Recurse on the child
                else {
                    TreeNode* childPtr = BuildNode(level + 1, childCoords); 
                    node->childList[index] = childPtr;
                    // Update the interior mask
                    if (childPtr != nullptr) {
                        node->interiorMask[q] |= (1 << r);
                        hasChild = true;
                    }
                }
            }
        }
    }
    // The node is empty
    if (!hasChild) {
        delete node;
        return nullptr;
    }

    ComputeCoords(node, coords);    
    if (node->level < m_voxels->gridLevels - 1) {
        ComputeInteriorMaskPC(node);
        CompactifyChildList(node);
    }
    return node;
}

uint32_t SceneBuilder::Index3D(uint32_t x, uint32_t y, uint32_t z, uint32_t dim) 
{
    return z + y * dim + x * dim * dim;    
}

void SceneBuilder::CountInteriorNodes(TreeNode* node) 
{
    m_voxels->interiorNodeCount[node->level]++;
 
    if (node->level < m_voxels->gridLevels - 1) {   
        uint32_t childCount = node->interiorMaskPC.back();
        for (uint32_t pos = 0; pos < childCount; pos++) {
            CountInteriorNodes(node->childList[pos]);
        }
    }
}

void SceneBuilder::AllocateStagingBuffers() 
{
    uint32_t tapeSize = m_voxels->tape.instructions.size() * sizeof(csg::Tape::Instr);
    uint32_t constantsSize = m_voxels->tape.constantPool.size() * sizeof(float);
    uint32_t nodeSize = 0;
    uint32_t childSize = 0;
    for (uint32_t level = 0; level < m_voxels->gridLevels; level++) {
        uint32_t dim = m_voxels->gridDims[level];
        nodeSize += m_voxels->interiorNodeCount[level] * m_voxels->NodeSize(level);
        // The last level nodes don't have any interior children :
        // they don't need any child list.
        if (level < m_voxels->gridLevels - 1) {
            childSize += m_voxels->interiorNodeCount[level] * (dim * dim * dim) * sizeof(uint32_t);
        }
    }

    m_stagingBuffers.node.Allocate(nodeSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_stagingBuffers.child.Allocate(childSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_stagingBuffers.tape.Allocate(tapeSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_stagingBuffers.constants.Allocate(constantsSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
}



void SceneBuilder::AllocateGPUBuffers() 
{
    // We assume the staging buffers were already allocated.
    VkBufferUsageFlags bufferUsage = 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    m_voxels->nodeBuffer.Allocate(m_stagingBuffers.node.size, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    m_voxels->childBuffer.Allocate(m_stagingBuffers.child.size, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    m_voxels->tapeBuffer.Allocate(m_stagingBuffers.tape.size, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    m_voxels->constPoolBuffer.Allocate(m_stagingBuffers.constants.size, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
}

uint32_t SceneBuilder::LayoutNode(TreeNode* node, std::vector<uint32_t>& nextNodeIdx) 
{
    // claim a node index
    uint32_t idx = nextNodeIdx[node->level]++;
    assert(idx < m_voxels->interiorNodeCount[node->level]);

    // Compute the address of the node in the staging buffer.
    size_t nodeAddr = (size_t)m_bufferContents.node;
    for (uint32_t i = 0; i < node->level; i++) {
        nodeAddr += m_voxels->interiorNodeCount[i] * m_voxels->NodeSize(i);
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
    memcpy((void*)(nodeAddr + m_voxels->NodeOfsInteriorMask(node->level)), 
        node->interiorMask.data(), 
        node->interiorMask.size() * sizeof(uint32_t));
    // Copy the mask PC
    memcpy((void*)(nodeAddr + m_voxels->NodeOfsInteriorMaskPC(node->level)),
        node->interiorMaskPC.data(),
        node->interiorMaskPC.size() * sizeof(uint32_t));
    // Copy the leaf mask
    memcpy((void*)(nodeAddr + m_voxels->NodeOfsLeafMask(node->level)),
        node->leafMask.data(),
        node->leafMask.size() * sizeof(uint32_t));

    if (node->level < m_voxels->gridLevels - 1) {
        // Compute the address of the node's child list in the child buffer.
        size_t childAddr = (size_t)m_bufferContents.child;
        for (uint32_t i = 0; i < node->level; i++) {
            childAddr += m_voxels->interiorNodeCount[i] * m_voxels->ChildListSize(i);
        }
        childAddr += idx * m_voxels->ChildListSize(node->level);
        uint32_t* childList = (uint32_t*)childAddr;

        // Recurse on children
        uint32_t childCount = node->interiorMaskPC.back();
        for (uint32_t pos = 0; pos < childCount; pos++) {
            childList[pos] = LayoutNode(node->childList[pos], nextNodeIdx);
        }
    }

    return idx;
}


void SceneBuilder::PrintVoxelStats()
{
    printf("[+] Grid dimensions : ");
    for (uint32_t dim : m_voxels->gridDims) {
        printf("%u ", dim);
    }
    printf("\n");

    uint32_t totalNodeCount = 0;
    for (uint32_t count : m_voxels->interiorNodeCount) {
        totalNodeCount += count;
    }
    printf("[+] Total interior nodes :\n\tcount=%u\tnode buf bytes=%lu\tchild buf bytes=%lu\n",
        totalNodeCount, m_stagingBuffers.node.size, m_stagingBuffers.child.size);

    printf("[+] Interior nodes per level :\n");
    for (uint32_t lvl = 0; lvl < m_voxels->gridLevels; lvl++) {
        uint32_t nodes = m_voxels->interiorNodeCount[lvl];
        uint32_t clBytes = (lvl < m_voxels->gridLevels - 1) ? 
            (nodes * m_voxels->ChildListSize(lvl)) : 0;

        printf("\tlevel %u:\tcount=%5u(%2.1f%%)\tnode buf bytes=%8u(%2.1f%%)\tchild buf bytes=%8u(%2.1f%%)\n", 
            lvl, 
            nodes, 100.0f * nodes / (float)totalNodeCount,
            nodes * m_voxels->NodeSize(lvl), 100.0f * nodes * m_voxels->NodeSize(lvl) / (float)m_stagingBuffers.node.size,
            clBytes, 100.0f * clBytes / (float)m_stagingBuffers.child.size);
    }
}

void SceneBuilder::ForEachTreeNode(std::function<void(TreeNode*)> f) 
{
    ForEachTreeNodeHelper(m_rootNode, f);    
}

void SceneBuilder::ForEachTreeNodeHelper(TreeNode* node, std::function<void(TreeNode*)> f) 
{
    if (node->level < m_voxels->gridLevels - 1) {
        uint32_t childCount = node->interiorMaskPC.back();
        for (uint32_t i = 0; i < childCount; i++) {
            ForEachTreeNodeHelper(node->childList[i], f);
        }
    }
    f(node);
}