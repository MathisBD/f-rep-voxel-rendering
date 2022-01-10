# f-rep-voxel-rendering
Large scale rendering of f-rep geometry, using sparse voxel datastructures, on the GPU.

# Building
Go to /build and run :
cmake -DCMAKE_BUILD_TYPE=Debug .. && make && ./proj

To print shader debug messages on stdout, run the following before launching the program :
export DEBUG_PRINTF_TO_STDOUT=true

# Sources

"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
Matthew J. Keeter, ACM Transactions on Graphics (Proceedings of SIGGRAPH), 2020

"Interactive k-D Tree GPU Raytracing"
Daniel Reiter Horn, Jeremy Sugerman, Mike Houston, Pat Hanrahan

# GPU data structures

GPU data layout : 4 large storage buffers

Interior Node Buffer : all interior nodes at level 0, then all at level 1, etc.
Tape buffer : the list of all tapes 

Interior Node structure :
    tape index        : 4 bytes (1 uint)
    coords            : 12 bytes (3 uints)
    leaf  mask        : res(L)^3 / 8 bytes
    interior mask     : res(L)^3 / 8 bytes 
    child list        : 4 * res(L)^3 bytes   
The coordinates of a node are given in the finest grid at this node's level.
The interior mask is the mask of the non-leaf children nodes.
The leaf mask is the mask of the leaf children nodes.
The leaf nodes don't occupy any space in memory,
    apart from the bit in the leaf mask. 
Child list : contains the indices (in the node buffer) of the non-leaf children nodes. 
    An index is 4 bytes. 
    There are res(L)^3 indices to store, where L is the level of the parent node.
Interior nodes at the maximum level don't have an interior mask or a child list.


# Voxelizer

The voxelizer builds a tree that isn't compactified, 
i.e. nodes can be empty or have all leaf children.
Whether the input to a particular stage is compactified or not should
however not make a difference.

Input : the tree nodes up to level i and the child lists up to level i-1.
Output : the tree nodes up to level i+1 and the child lists up to level i.

Thread group size : (dim[i+1], dim[i+1], dim[i+1]).
Number of groups to spawn : (n_nodes[i], 1, 1).
Thread (x, y, z) in group k handles the child at (x, y , z) in the node k of level i.

On the CPU : allocate enough space for n_nodes[i] * dim[i+1]**3 nodes and child lists.
    For now even empty nodes will have a child list.

On the GPU : a global atomic counters (starting at 0)
    claims nodes in the node buffer at level i+1.

Step 1: interval evaluation, using the parent tape.

Step 2.1: if child is an empty node, do nothing.
Step 2.2: if child is a leaf node, update the parent leaf mask.
Step 2.3: if child is an interior node (interval evaluation was ambiguous), 
    update parent interior node mask.

if (level == maxLevel) then all threads finish now.

THREAD BARRIER : we need the interior mask in the parent to be complete
Step 3: in parallel over all threads in the work group,
    calculate the interior mask PC of the parent.

THREAD BARRIER : we need the total interior node count in the parent, 
    and thus the last entry of mask PC
Step 4.1: thread (0, 0, 0) claims an index range of size n_interior_nodes(this thread group) 
    in the node buffer at level i+1.
    The start of the range (base_idx) is stored in shared memory.
THREAD BARRIER : we need the shared index.
Step 4.2: each thread with an interior node:
    -> claims an index in the node buffer at level i+1 
        (using base_idx + cl_idx, no need for an atomic here).
    -> fills in the basic info for the child node : tape idx, coords
Step 4.3: each thread with an interior node:
    -> calculates the corresponding child list index (using the parent interior mask and maskPC).
    -> stores the node id in the parent child list
    
# Compactifier