# f-rep-voxel-rendering
Large scale rendering of f-rep geometry, using sparse voxel datastructures, on the GPU.

# Sources

"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces"
Matthew J. Keeter, ACM Transactions on Graphics (Proceedings of SIGGRAPH), 2020

# Tmp

Sparse hierarchy of grids. The resolution res(L) of a level is the log_2 of the level's dimension.

GPU data layout : 3 large storage buffers

Node Buffer  : nodes for all levels
    -> all nodes at level 0, then all at level 1, etc.
Child Buffer : child lists for all levels
    -> all lists at level 0, then all at level 1, etc.
Voxel Buffer : voxel data

Node structure :
    child list index : 4 bytes
    cell mask        : res(L)^3 / 8 bytes
    cell mask PC     : res(L)^3 / 8 bytes    

Child list : contains the indices (in the node buffer) of the children. If the node is a leaf node, then the indices instead refer to the voxels of the node.
    An index is 4 bytes. 
    There are res(L)^3 indices to store, where L is the level of the parent node.

Voxel buffer : list of all voxels. Not ordered by level.
Voxel structure :
    normal :         3 floats (12 bytes)
    material index : 1 uint   (4 bytes)
    --> material colors are stored in the uniform buffer