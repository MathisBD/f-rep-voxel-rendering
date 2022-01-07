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
    m_stagingBuffers.voxel.Init(m_allocator);
    m_stagingBuffers.tape.Init(m_allocator);
    m_stagingBuffers.constants.Init(m_allocator);
}

void SceneBuilder::Cleanup() 
{
    m_stagingBuffers.node.Cleanup();
    m_stagingBuffers.child.Cleanup();
    m_stagingBuffers.voxel.Cleanup();
    m_stagingBuffers.tape.Cleanup();
    m_stagingBuffers.constants.Cleanup();
}

void SceneBuilder::BuildScene() 
{
    m_rootNode = BuildNode(0, { 0, 0, 0 });
    // The scene shouldn't be empty.
    assert(m_rootNode);

    CountNodes(m_rootNode);
    AllocateStagingBuffers();

    m_bufferContents.node = m_stagingBuffers.node.Map(); 
    m_bufferContents.child = m_stagingBuffers.child.Map(); 
    m_bufferContents.voxel = m_stagingBuffers.voxel.Map(); 
    m_bufferContents.tape = m_stagingBuffers.tape.Map();
    m_bufferContents.constants = m_stagingBuffers.constants.Map();

    std::vector<uint32_t> nextNodeIdx(m_voxels->gridLevels, 0);
    LayoutNode(m_rootNode, nextNodeIdx);
    // layout the voxels
    memcpy(m_bufferContents.voxel, m_voxelData.data(), 
        m_voxelData.size() * sizeof(VoxelStorage::Voxel));
    // layout the tape
    memcpy(m_bufferContents.tape, m_voxels->tape.instructions.data(), 
        m_voxels->tape.instructions.size() * sizeof(csg::Tape::Instr));
    // layout the tape constants
    memcpy(m_bufferContents.constants, m_voxels->tape.constantPool.data(),
        m_voxels->tape.constantPool.size() * sizeof(float));

    m_stagingBuffers.node.Unmap();
    m_stagingBuffers.child.Unmap();
    m_stagingBuffers.voxel.Unmap();
    m_stagingBuffers.tape.Unmap();
    m_stagingBuffers.constants.Unmap();

    // Print some stats before we cleanup.
    PrintVoxelStats();

    // We can already delete the nodes and voxel vector.
    std::vector<VoxelStorage::Voxel> empty;
    m_voxelData.swap(empty);

    DeleteNode(m_rootNode);
    m_rootNode = nullptr;
}

void SceneBuilder::DeleteNode(TreeNode* node) 
{
    if (node->level < m_voxels->gridLevels - 1) {
        uint32_t childCount = node->maskPC.back();
        for (uint32_t i = 0; i < childCount; i++) {
            DeleteNode((TreeNode*)node->childList[i]);
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

    VkBufferCopy voxelRegion;
    voxelRegion.srcOffset = 0;
    voxelRegion.dstOffset = 0;
    voxelRegion.size = m_stagingBuffers.voxel.size;
    vkCmdCopyBuffer(cmd, m_stagingBuffers.voxel.buffer, m_voxels->voxelBuffer.buffer, 1, &voxelRegion); 

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
    float d = m_voxels->tape.Eval(pos.x, pos.y, pos.z);

    if (d < 0.0f) {
        return std::numeric_limits<size_t>::max();
    }

    VoxelStorage::Voxel voxel;
    /*float eps = 0.0001f;
    voxel.normal = -glm::normalize(glm::vec3(
        (m_voxels->tape.Eval(pos.x + eps, pos.y, pos.z) - d) / eps,
        (m_voxels->tape.Eval(pos.x, pos.y + eps, pos.z) - d) / eps,
        (m_voxels->tape.Eval(pos.x, pos.y, pos.z + eps) - d) / eps ));*/
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
    // Check there is at least one uint in the mask.
    // We could treat this as a special case but it would make
    // calculating the node size/offsets more complex for no real benefit.
    assert((dim*dim*dim) / 32 > 0);
    node->mask = std::vector<uint32_t>((dim*dim*dim) / 32);
    node->maskPC = std::vector<uint32_t>((dim*dim*dim) / 32);
    node->childList = std::vector<size_t>(dim*dim*dim);

    // Recurse on the children / create the voxels
    for (uint32_t x = 0; x < dim; x++) {
        for (uint32_t y = 0; y < dim; y++) {
            for (uint32_t z = 0; z < dim; z++) {
                uint32_t index = Index3D(x, y, z, dim);
                glm::u32vec3 childCoords = { x, y, z };
                childCoords += coords * m_voxels->gridDims[level];

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
        delete node;
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
    uint32_t tapeSize = m_voxels->tape.instructions.size() * sizeof(csg::Tape::Instr);
    uint32_t constantsSize = m_voxels->tape.constantPool.size() * sizeof(float);
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
    m_voxels->voxelBuffer.Allocate(m_stagingBuffers.voxel.size, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);    
    m_voxels->tapeBuffer.Allocate(m_stagingBuffers.tape.size, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    m_voxels->constPoolBuffer.Allocate(m_stagingBuffers.constants.size, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
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
        childAddr += m_voxels->nodeCount[i] * glm::pow(m_voxels->gridDims[i], 3) * sizeof(uint32_t);
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

        /*printf("%s:%u level=%d\tcoords=%u %u %u\n", 
            node->level == m_voxels->gridLevels - 1 ? "LEAF" : "INTERIOR",
            idx, node->level, coords[0], coords[1], coords[2]);
        printf("\tmask = ");
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
        printf("\n");

        printf("\tchildren = ");
        for (size_t i = 0; i < node->maskPC.back(); i++) {
            printf("%u ", childList[i]);
        }
        printf("\n\n");*/
    }

    return idx;
}


void SceneBuilder::PrintVoxelStats()
{
    printf("[+] Total voxels :\n\tcount=%lu\tvoxel buf bytes=%lu\n", 
        m_stagingBuffers.voxel.size / sizeof(VoxelStorage::Voxel),
        m_stagingBuffers.voxel.size);

    uint32_t totalNodeCount = 0;
    for (uint32_t count : m_voxels->nodeCount) {
        totalNodeCount += count;
    }
    printf("[+] Total nodes :\n\tcount=%u\tnode buf bytes=%lu\tchild buf bytes=%lu\n",
        totalNodeCount, m_stagingBuffers.node.size, m_stagingBuffers.child.size);

    printf("[+] Nodes per level :\n");
    for (uint32_t lvl = 0; lvl < m_voxels->gridLevels; lvl++) {
        uint32_t nodes = m_voxels->nodeCount[lvl];
        printf("\tlevel %u:\tcount=%5u(%2.1f%%)\tnode buf bytes=%8u(%2.1f%%)\tchild buf bytes=%8u(%2.1f%%)\n", 
            lvl, 
            nodes, 100.0f * nodes / (float)totalNodeCount,
            nodes * m_voxels->NodeSize(lvl), 100.0f * nodes * m_voxels->NodeSize(lvl) / (float)m_stagingBuffers.node.size,
            nodes * m_voxels->ChildListSize(lvl), 100.0f * nodes * m_voxels->ChildListSize(lvl) / (float)m_stagingBuffers.child.size);
    }

    // childListFullness[lvl][i] contains the number of nodes on lvl
    // whose child count is smaller than ((i+1) / buckets) times the max count.
    uint32_t buckets = 10;
    std::vector<std::vector<uint32_t>> childListFullness(m_voxels->gridLevels);
    for (uint32_t lvl = 0; lvl < m_voxels->gridLevels; lvl++) {
        childListFullness[lvl] = std::vector<uint32_t>(buckets);
    }
    std::vector<uint32_t> fullNodes(m_voxels->gridLevels, 0); 
    
    ForEachTreeNode([&] (TreeNode* node) {
        uint32_t childCount = node->maskPC.back();
        uint32_t maxChildCount = glm::pow(m_voxels->gridDims[node->level], 3);
        assert(childCount <= maxChildCount);
        if (childCount == maxChildCount) {
            fullNodes[node->level]++;
        }
        for (uint32_t i = 0; i < buckets; i++) {
            if (childCount / (float)maxChildCount <= (i+1) / (float)buckets) {
                childListFullness[node->level][i]++;
            }
        }
    });
    
    printf("[+] Node stats per level :\n");
    for (uint32_t lvl = 0; lvl < m_voxels->gridLevels; lvl++) {
        printf("\tlevel=%u: full nodes=%2.1f%%\t", 
            lvl, 100.0f * fullNodes[lvl] / (float)m_voxels->nodeCount[lvl]);
        printf("child list fullness=");
        for (uint32_t i = 0; i < buckets; i++) {
            printf("%2.1f%%  ", 100.0f * childListFullness[lvl][i] / (float)m_voxels->nodeCount[lvl]);
        }
        printf("\n");
    }
}

void SceneBuilder::ForEachTreeNode(std::function<void(TreeNode*)> f) 
{
    ForEachTreeNodeHelper(m_rootNode, f);    
}

void SceneBuilder::ForEachTreeNodeHelper(TreeNode* node, std::function<void(TreeNode*)> f) 
{
    if (node->level < m_voxels->gridLevels - 1) {
        uint32_t childCount = node->maskPC.back();
        for (uint32_t i = 0; i < childCount; i++) {
            ForEachTreeNodeHelper((TreeNode*)node->childList[i], f);
        }
    }
    f(node);
}