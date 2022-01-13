#include "engine/scene_builder.h"
#include <string.h>


void SceneBuilder::Init(VmaAllocator vmaAllocator, VoxelStorage* voxels) 
{
    m_allocator = vmaAllocator;
    m_voxels = voxels;
    
    m_stagingBuffers.node.Init(m_allocator);
    m_stagingBuffers.tape.Init(m_allocator);
}

void SceneBuilder::Cleanup() 
{
    m_stagingBuffers.node.Cleanup();
    m_stagingBuffers.tape.Cleanup();
    m_threadPool.Stop();
}

void SceneBuilder::BuildScene() 
{
    m_rootNode = BuildNode(0, { 0, 0, 0 });
    // The scene shouldn't be empty.
    assert(m_rootNode);

    CountInteriorNodes(m_rootNode);
    AllocateStagingBuffers();

    m_bufferContents.node = m_stagingBuffers.node.Map(); 
    m_bufferContents.tape = m_stagingBuffers.tape.Map();

    std::vector<uint32_t> nextNodeIdx(m_voxels->gridLevels, 0);
    LayoutNode(m_rootNode, nextNodeIdx);
    // layout the tape
    memcpy(m_bufferContents.tape, m_voxels->tape.instructions.data(), 
        m_voxels->tape.instructions.size() * sizeof(csg::Tape::Instr));
    
    m_stagingBuffers.node.Unmap();
    m_stagingBuffers.tape.Unmap();

    // Print some stats before we cleanup.
    PrintVoxelStats();

    // We can already delete the nodes.
    DeleteNode(m_rootNode);
    m_rootNode = nullptr;
}

void SceneBuilder::DeleteNode(TreeNode* node) 
{
    if (node->level < m_voxels->gridLevels - 1) {
        for (TreeNode* child : node->childList) {
            if (child != nullptr) {
                DeleteNode(child);
            }
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

    VkBufferCopy tapeRegion;
    tapeRegion.srcOffset = 0;
    tapeRegion.dstOffset = 0;
    tapeRegion.size = m_stagingBuffers.tape.size;
    vkCmdCopyBuffer(cmd, m_stagingBuffers.tape.buffer, m_voxels->tapeBuffer.buffer, 1, &tapeRegion);
}

bool SceneBuilder::HasAllLeafChildren(TreeNode* node) 
{
    uint32_t dim = m_voxels->gridDims[node->level];
    for (uint32_t i = 0; i < (dim*dim*dim) / 32; i++) {
        if (node->leafMask[i] != 0xFFFFFFFF) {
            return false;
        }
    }    
    return true;
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
                    float density = m_voxels->tape.Eval(pos.x, pos.y, pos.z, 0);
                    // Simply update the leaf mask
                    if (density < 0.0f) {
                        node->leafMask[q] |= (1 << r);
                        hasChild = true;
                    }
                }
                // Recurse on the child
                else {
                    TreeNode* childPtr = BuildNode(level + 1, childCoords); 
                    // Add a leaf node child
                    if (childPtr != nullptr && HasAllLeafChildren(childPtr)) {
                        delete childPtr;
                        node->leafMask[q] |= (1 << r);
                        hasChild = true;
                    }   
                    // Add an interior node child
                    else if (childPtr != nullptr) {
                        node->childList[index] = childPtr;
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
        for (TreeNode* child : node->childList) {
            if (child != nullptr) {
                CountInteriorNodes(child);
            }
        }
    }
}

void SceneBuilder::AllocateStagingBuffers() 
{
    uint32_t tapeSize = m_voxels->tape.instructions.size() * sizeof(csg::Tape::Instr);
    uint32_t nodeSize = 0;
    for (uint32_t level = 0; level < m_voxels->gridLevels; level++) {
        nodeSize += m_voxels->interiorNodeCount[level] * m_voxels->NodeSize(level);
    }
    m_stagingBuffers.node.Allocate(nodeSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_stagingBuffers.tape.Allocate(tapeSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
}



void SceneBuilder::AllocateGPUBuffers() 
{
    // We assume the staging buffers were already allocated.
    m_voxels->nodeBuffer.Allocate(m_stagingBuffers.node.size,   
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VMA_MEMORY_USAGE_GPU_ONLY);
    m_voxels->tapeBuffer.Allocate(m_stagingBuffers.tape.size,   
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VMA_MEMORY_USAGE_GPU_ONLY);
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

    // Copy the leaf mask
    memcpy((void*)(nodeAddr + m_voxels->NodeOfsLeafMask(node->level)),
        node->leafMask.data(),
        node->leafMask.size() * sizeof(uint32_t));
    
    if (node->level < m_voxels->gridLevels - 1) {
        // Copy the interior mask
        memcpy((void*)(nodeAddr + m_voxels->NodeOfsInteriorMask(node->level)), 
            node->interiorMask.data(), 
            node->interiorMask.size() * sizeof(uint32_t));
        
        // Create the child list
        std::vector<uint32_t> childList;
        for (TreeNode* child : node->childList) {
            if (child != nullptr) {
                childList.push_back(LayoutNode(child, nextNodeIdx));
            }
            else {
                childList.push_back(0);
            }
        }
        // Copy the child list
        memcpy((void*)(nodeAddr + m_voxels->NodeOfsChildList(node->level)),
            childList.data(), childList.size() * sizeof(uint32_t));
    }
    return idx;
}


void SceneBuilder::PrintVoxelStats()
{
    printf("[+] Grid dimensions : ");
    for (uint32_t dim : m_voxels->gridDims) {
        printf("%u ", dim);
    }
    printf("  total = %u\n", m_voxels->fineGridDim);

    uint32_t totalNodeCount = 0;
    for (uint32_t count : m_voxels->interiorNodeCount) {
        totalNodeCount += count;
    }
    printf("[+] Total interior nodes :\n\tcount=%u\tnode buf bytes=%lu\n",
        totalNodeCount, m_stagingBuffers.node.size);

    printf("[+] Interior nodes per level :\n");
    for (uint32_t lvl = 0; lvl < m_voxels->gridLevels; lvl++) {
        uint32_t nodes = m_voxels->interiorNodeCount[lvl];
        
        printf("\tlevel %u:\tcount=%5u(%2.1f%%)\tnode buf bytes=%8u(%2.1f%%)\n", 
            lvl, 
            nodes, 100.0f * nodes / (float)totalNodeCount,
            nodes * m_voxels->NodeSize(lvl), 100.0f * nodes * m_voxels->NodeSize(lvl) / (float)m_stagingBuffers.node.size);
    }

    uint32_t buckets = 10;
    std::vector<uint32_t> childListFullness(buckets, 0);
    uint32_t clNodes = 0;

    ForEachTreeNode([&] (TreeNode* node) {
        if (node->level < m_voxels->gridLevels - 1) {
            uint32_t childCount = 0;
            uint32_t maxChildCount = 0;
            for (TreeNode* child : node->childList) {
                if (child != nullptr) {
                    childCount++;
                }
                maxChildCount++;
            }
            assert(maxChildCount == glm::pow(m_voxels->gridDims[node->level], 3));
            
            for (uint32_t i = 0; i < buckets; i++) {
                if (childCount / (float)maxChildCount <= (i+1) / (float)buckets) {
                    childListFullness[i]++;
                }
            }
            clNodes++;
        }
    });

    printf("[+] Child list fullness :\n");
    for (uint32_t i = 0; i < buckets; i++) {
        printf("\tUp to %.2f%%: %.2f%%\n", 
            100.0f * (i+1) / (float)buckets, 
            100.0f * childListFullness[i] / (float)clNodes);
    }
    printf("\n");
}

void SceneBuilder::ForEachTreeNode(std::function<void(TreeNode*)> f) 
{
    ForEachTreeNodeHelper(m_rootNode, f);    
}

void SceneBuilder::ForEachTreeNodeHelper(TreeNode* node, std::function<void(TreeNode*)> f) 
{
    if (node->level < m_voxels->gridLevels - 1) {
        for (TreeNode* child : node->childList) {
            if (child != nullptr) {
                ForEachTreeNodeHelper(child, f);
            }
        }
    }
    f(node);
}