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
Child Buffer : child lists for all interior nodes
    -> all lists at level 0, then all at level 1, etc.
Tape buffer : the list of all tapes 

Interior Node structure :
    child list index  : 4 bytes (1 uint)
    tape index        : 4 bytes (1 uint)
    coordinates       : 12 bytes (3 uints)
    interior mask     : res(L)^3 / 8 bytes
    interior mask PC  : res(L)^3 / 8 bytes
    leaf  mask        : res(L)^3 / 8 bytes    
The coordinates of a node are given in the finest grid.
The interior mask is the mask of the non-leaf children nodes.
The leaf mask is the mask of the leaf children nodes.

The leaf nodes don't occupy any space in memory.


Child list : contains the indices (in the node buffer) of the non-leaf children nodes. 
    An index is 4 bytes. 
    There are res(L)^3 indices to store, where L is the level of the parent node.
