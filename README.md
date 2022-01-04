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
    coordinates      : 12 bytes (3 uints)
    cell mask        : res(L)^3 / 8 bytes
    cell mask PC     : res(L)^3 / 8 bytes    
The coordinates of a node are given in the finest grid.


Child list : contains the indices (in the node buffer) of the children. If the node is a leaf node, then the indices instead refer to the voxels of the node.
    An index is 4 bytes. 
    There are res(L)^3 indices to store, where L is the level of the parent node.

Voxel buffer : list of all voxels. Not ordered by level.
Voxel structure :
    normal :         3 floats (12 bytes)
    material index : 1 uint   (4 bytes)
    --> material colors are stored in the uniform buffer