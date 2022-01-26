#version 450

#extension GL_EXT_debug_printf : enable

// Include other files  
//#define DEBUG_LOG
const float infinity = 1. / 0.;
const float pi = 3.141592653589793238462643383279502884197;

#define CUBE(x) ((x)*(x)*(x))

#define INDEX3D(cell, dim) ((cell).z + (dim) * (cell).y + (dim) * (dim) * (cell).x)

#define IS_BIT_SET(x, bit) ((x) & (1 << (bit)))

// returns the smallest multiple of k that is 
// greater or equal to n.
#define ALIGN(n, k) ((((n) + (k) - 1) / (k)) * (k))

#define MAX_3(x, y, z) max((x), max((y), (z)))
#define MIN_3(x, y, z) min((x), min((y), (z)))
#define MAX_4(x, y, z, w) max(max((x), (y)), max((z), (w)))
#define MIN_4(x, y, z, w) min(min((x), (y)), min((z), (w)))

#define EPS 0.0001f

// for each component of v : 
// returns 1.0f if it is >= 0, and 0.0f otherwise.
#define SIGN_01(v) vec3(greaterThanEqual((v), vec3(0, 0, 0)))

// 'array' is an array of uints, and the bitmask starts at array[ofs]. 
#define BITSET_SET(array, ofs, i)          array[(ofs) + ((i) >> 5)] |= (1 << ((i) & 0x1F))
#define BITSET_ATOMIC_SET(array, ofs, i)   atomicOr(array[(ofs) + ((i) >> 5)], (1 << ((i) & 0x1F)))

#define BITSET_CLEAR(array, ofs, i)        array[(ofs) + ((i) >> 5)] &= ~(1 << ((i) & 0x1F))
#define BITSET_ATOMIC_CLEAR(array, ofs, i) atomicAnd(array[(ofs) + ((i) >> 5)], ~(1 << ((i) & 0x1F)))

#define BITSET_LOAD(array, ofs, i)         bool(array[(ofs) + ((i) >> 5)] & (1 << ((i) & 0x1F)))

#define ZERO_ARRAY(array, ofs, size) for (uint i = 0; i < (size); i++) { array[(ofs) + i] = 0; }

#ifdef DEBUG_LOG

#define SHOULD_LOG() (gl_GlobalInvocationID.xyz == vec3(1, 3, 2))

#define LOG0(fmt) if (SHOULD_LOG()) {\
    debugPrintfEXT((fmt)); }  

#define LOG1(fmt, x1) if (SHOULD_LOG()) {\
    debugPrintfEXT((fmt), (x1)); }  

#define LOG2(fmt, x1, x2) if (SHOULD_LOG()) {\
    debugPrintfEXT((fmt), (x1), (x2)); }  

#define LOG3(fmt, x1, x2, x3) if (SHOULD_LOG()) {\
    debugPrintfEXT((fmt), (x1), (x2), (x3)); }  

#define LOG4(fmt, x1, x2, x3, x4) if (SHOULD_LOG()) {\
    debugPrintfEXT((fmt), (x1), (x2), (x3), (x4)); }  

#define LOG5(fmt, x1, x2, x3, x4, x5) if (SHOULD_LOG()) {\
    debugPrintfEXT((fmt), (x1), (x2), (x3), (x4), (x5)); }  

#define LOG6(fmt, x1, x2, x3, x4, x5, x6) if (SHOULD_LOG()) {\
    debugPrintfEXT((fmt), (x1), (x2), (x3), (x4), (x5), (x6)); }  

#define LOG7(fmt, x1, x2, x3, x4, x5, x6, x7) if (SHOULD_LOG()) {\
    debugPrintfEXT((fmt), (x1), (x2), (x3), (x4), (x5), (x6), (x7)); }  

#define LOG8(fmt, x1, x2, x3, x4, x5, x6, x7, x8) if (SHOULD_LOG()) {\
    debugPrintfEXT((fmt), (x1), (x2), (x3), (x4), (x5), (x6), (x7), (x8)); }  

#else

#define LOG0(fmt)                                   do {} while(false)
#define LOG1(fmt, x1)                               do {} while(false)
#define LOG2(fmt, x1, x2)                           do {} while(false)
#define LOG3(fmt, x1, x2, x3)                       do {} while(false)
#define LOG4(fmt, x1, x2, x3, x4)                   do {} while(false)
#define LOG5(fmt, x1, x2, x3, x4, x5)               do {} while(false)
#define LOG6(fmt, x1, x2, x3, x4, x5, x6)           do {} while(false) 
#define LOG7(fmt, x1, x2, x3, x4, x5, x6, x7)       do {} while(false)
#define LOG8(fmt, x1, x2, x3, x4, x5, x6, x7, x8)   do {} while(false) 

#endif

#define THREAD_GROUP_SIZE_X 16
#define THREAD_GROUP_SIZE_Y 16
// The valid levels range from 0 to LEVEL_COUNT-1 included.
// Level 0 contains the unique root node, and LEVEL_COUNT-1 the last interior nodes,
// whose only children are leafs (and empty nodes).
// The maximum number of tape slots.
#define MAX_SLOT_COUNT 128
#define MAX_CONST_POOL_SIZE 256
#define MAX_LIGHT_COUNT 8
#define LEVEL_COUNT 4

// Kernel group size.
layout (
    local_size_x = THREAD_GROUP_SIZE_X, 
    local_size_y = THREAD_GROUP_SIZE_Y) in;
 
#define MAX_LEVEL_COUNT 8

struct VoxelData {
    vec3 normal;
    uint material_idx;
};

struct DirectionalLight {
    vec4 color;
    vec4 direction;
};

struct LevelData {
    uint dim;
    // Offsets in the node buffers.
    // It is given in uints rather than in bytes.
    uint node_ofs;
    // The world size of a single child cell at this level.
    // The cell_size of the last level (level_count-1) is the voxel size.
    // The cell_size of level 0 is world_size / dim[0].
    float cell_size;
    uint _padding_;
};

// Output image.
layout (set = 0, binding = 0, rgba8) uniform image2DArray out_img;

// Shader basic data.
layout (set = 0, binding = 1) uniform ParamsBuffer {
    // The actual time of this shader's dispatch,
    // used e.g. to seed random functions.
    float time;
    // the time used to evaluate the tape (and thus the normals).
    float tape_time; 
    // which image we are going to right to in the image array
    // (for temporal supersampling).
    uint out_img_layer;
    uint light_count;

    vec4 camera_pos;
    vec4 camera_forward;
    vec4 camera_up;
    vec4 camera_right;

    vec3 grid_world_pos;
    float grid_world_size;
    
    uvec2 screen_res;
    vec2 screen_world_size;
    vec4 background_color;

    DirectionalLight lights[MAX_LIGHT_COUNT]; 
    // The tape constant pool.
    vec4 const_pool[MAX_CONST_POOL_SIZE / 4];
    LevelData levels[MAX_LEVEL_COUNT];
} params_buf;

// We need the std430 layout here so that the array stride is 4 bytes.
// The std140 layout would round it up to 16 bytes.
layout (std430, set = 0, binding = 2) readonly buffer NodeBuffer {
    uint data[];
} node_buf;

layout (std430, set = 0, binding = 3) readonly buffer TapeBuffer {
    uint data[];
} tape_buf;



// The maximum number of min/max instructions per tape.
#define MAX_MM_OPS          MAX_TAPE_SIZE
// When doing interval evaluation,
// this defines the size of the array that holds
// the configurations for inputs of min/max ops.
// Each configuration is 2 bits (because there are 4 possibilities).
#define MM_ARRAY_SIZE       (MAX_MM_OPS / 16)

#define MM_NONE   0
#define MM_FIRST  1
#define MM_SECOND 2
#define MM_BOTH   (MM_FIRST | MM_SECOND)

#define OP_CONSTANT 0
#define OP_SIN      1
#define OP_COS      2
#define OP_ADD      3
#define OP_SUB      4
#define OP_MUL      5
#define OP_DIV      6
#define OP_MIN      7
#define OP_MAX      8
#define OP_EXP      9
#define OP_NEG      10
#define OP_SQRT     11
#define OP_COPY     12


// An unpacked tape instruction
struct Instruction {
    uint op;
    uint outSlot;
    uint inSlotA;
    uint inSlotB;
};


// Returns the number of instructions in the tape.
uint tape_read_size(uint tape_idx)
{
    return tape_buf.data[tape_idx];
}

Instruction tape_read_inst(uint tape, uint idx)
{
    // Add 1 to skip the tape size.
    uint data = tape_buf.data[tape + idx + 1];

    Instruction inst;
    inst.op      = (data >> 0)  & 0xFF;
    inst.outSlot = (data >> 8)  & 0xFF;
    inst.inSlotA = (data >> 16) & 0xFF;
    inst.inSlotB = (data >> 24) & 0xFF;
    return inst;
}


// Evaluate the gradient of the tape function
// using forward mode automatic differentiation.
vec3 tape_eval_gradient(uint tape, vec3 pos)
{
    // A slot contains :
    // x, y, z : partial derivatives wrt. x, y and z
    // w : the value of the function
    vec4 slots[MAX_SLOT_COUNT];
    slots[0] = vec4(1, 0, 0, pos.x);
    slots[1] = vec4(0, 1, 0, pos.y);
    slots[2] = vec4(0, 0, 1, pos.z);
    slots[3] = vec4(0, 0, 0, params_buf.tape_time);

    uint idx = 0;
    uint size = tape_read_size(tape);
    while (idx < size) {
        // Decode the instruction
        Instruction i = tape_read_inst(tape, idx);
        vec4 a = slots[i.inSlotA];
        vec4 b = slots[i.inSlotB];

        // Dispatch        
        switch (i.op) {
        case OP_CONSTANT:
            slots[i.outSlot] = vec4(0, 0, 0, params_buf.const_pool[i.inSlotA / 4][i.inSlotA % 4]);
            break;
        case OP_SIN:
            slots[i.outSlot] = vec4(
                cos(a.w) * a.xyz,
                sin(a.w));
            break;
        case OP_COS:
            slots[i.outSlot] = vec4(
                -sin(a.w) * a.xyz,
                cos(a.w));
            break;
        case OP_ADD:
            slots[i.outSlot] = a + b;
            break;
        case OP_SUB:
            slots[i.outSlot] = a - b;
            break;
        case OP_MUL:
            slots[i.outSlot] = vec4(
                a.xyz * b.w + a.w * b.xyz,
                a.w * b.w);
            break;
        case OP_DIV:
            slots[i.outSlot] = vec4(
                (a.xyz * b.w - a.w * b.xyz) / (b.w*b.w),
                a.w / b.w);
            break;
        case OP_MIN:
            if (a.w < b.w) {
                slots[i.outSlot] = a;
            }
            else {
                slots[i.outSlot] = b;
            }       
            break;
        case OP_MAX:
            if (a.w > b.w) {
                slots[i.outSlot] = a;
            }
            else {
                slots[i.outSlot] = b;
            }
            break;
        case OP_EXP:
            slots[i.outSlot] =  exp(a.w) * vec4(a.xyz, 1);
            break;
        case OP_NEG:
            slots[i.outSlot] = -a;       
            break;
        case OP_SQRT:
            slots[i.outSlot] = vec4(
                a.xyz / (2 * sqrt(a.w)),
                sqrt(a.w));       
            break;
        case OP_COPY:
            slots[i.outSlot] = a;       
            break;
        } 
        idx++;
    }   
    return slots[0].xyz;
}


// Node structure.
// Sizes are given in uints rather than bytes.
#define NODE_SIZE(level)                                    \
    ((level) == LEVEL_COUNT - 1 ?                \
    4 + (CUBE(params_buf.levels[(level)].dim) >> 4) :       \
    4 + 17 * (CUBE(params_buf.levels[(level)].dim) >> 4))

#define NODE_OFS_TAPE_IDX(level)         0
#define NODE_OFS_COORDS(level)           1
#define NODE_OFS_LEAF_MASK(level)        4 
// Nodes on level (params_buf.level_count - 1) don't have
// an interior mask or a child list.
#define NODE_OFS_INTERIOR_MASK(level)    (4 + (CUBE(params_buf.levels[(level)].dim) >> 5))
#define NODE_OFS_CHILD_LIST(level)       (4 + (CUBE(params_buf.levels[(level)].dim) >> 4))


// child_index == INDEX(local position in the node)
bool node_has_leaf_child(uint node_id, uint level, uint child_index)
{
    // get the leaf mask position in the node buffer
    uint mask_pos = 
        params_buf.levels[level].node_ofs +
        NODE_SIZE(level) * node_id +
        NODE_OFS_LEAF_MASK(level);
    return BITSET_LOAD(node_buf.data, mask_pos, child_index);
}

// child_index == INDEX(local position in the node)
bool node_has_interior_child(uint node_id, uint level, uint child_index)
{
    // Make sure we don't read from a non-existant interior mask.
    if (level >= LEVEL_COUNT - 1) {
        return false;
    }

    // The node's interior mask position in the node buffer.
    uint mask_pos = 
        params_buf.levels[level].node_ofs +
        NODE_SIZE(level) * node_id +
        NODE_OFS_INTERIOR_MASK(level);
    return BITSET_LOAD(node_buf.data, mask_pos, child_index);
}

// child_index == INDEX(local position in the node)
// Returns the node id of the child.
uint node_get_interior_child(uint node_id, uint level, uint child_index)
{
    // The node's child list position in the node buffer.
    uint child_list_pos = 
        params_buf.levels[level].node_ofs +
        NODE_SIZE(level) * node_id +
        NODE_OFS_CHILD_LIST(level);
    return node_buf.data[child_list_pos + child_index];
}

uint node_get_tape_index(uint node, uint level)
{
    uint idx_pos = params_buf.levels[level].node_ofs + 
        node * NODE_SIZE(level) + NODE_OFS_TAPE_IDX(level);
    return node_buf.data[idx_pos];
}

uvec3 node_get_coords(uint node, uint level)
{
    uint coords_pos = params_buf.levels[level].node_ofs +
        node * NODE_SIZE(level) + NODE_OFS_COORDS(level);
    return uvec3(
        node_buf.data[coords_pos + 0],
        node_buf.data[coords_pos + 1],
        node_buf.data[coords_pos + 2]);
}


float _rand_seed;

void random_seed(float seed)
{
    _rand_seed = seed;
}

float rand()
{
    float result = fract(sin(_rand_seed / 100.0f * dot(gl_GlobalInvocationID.xy, vec2(12.9898f, 78.233f))) * 43758.5453f);
    _rand_seed += 1.0f;
    return result;
}


struct Hit {
    float t;
    uint node;
    uint level;
    uint cell_index;
};


// Returns true if there is an intersection.
// A positive value t means the intersection point is at orig + t * dir.
bool raygrid_intersect(vec3 orig, vec3 dir, out float t_enter, out float t_leave)
{
    vec3 invDir = 1.0f / dir;
    vec3 t_0 = (params_buf.grid_world_pos + vec3(EPS, EPS, EPS) - orig) * invDir; 
    vec3 t_1 = (params_buf.grid_world_pos + params_buf.grid_world_size - vec3(EPS, EPS, EPS) - orig) * invDir;
    
#define SWAP(a, b) { float tmp = a; a = b; b = tmp; }
    // We chech invDir.x < 0f rather than t_0.x > t_1.x (almost equivalent),
    // because of numerical stability issues when dir.x == 0f.
    // See the comment in raytrace() for an explanation.
    if (invDir.x < 0.0f) SWAP(t_0.x, t_1.x)
    if (invDir.y < 0.0f) SWAP(t_0.y, t_1.y)
    if (invDir.z < 0.0f) SWAP(t_0.z, t_1.z)
#undef SWAP

    t_enter = MAX_3(t_0.x, t_0.y, t_0.z);
    t_leave = MIN_3(t_1.x, t_1.y, t_1.z);

    return t_enter <= t_leave && t_leave >= 0.0f;
}


bool dda(vec3 orig, vec3 dir, out Hit hit)
{
    // Check the ray intersects the root node.
    float grid_t_enter;
    float grid_t_leave;
    bool hit_grid = raygrid_intersect(orig, dir, grid_t_enter, grid_t_leave);
    if (!hit_grid) {
        return false;
    }

    // DDA parameters. 
    vec3 invDir = 1.0f / dir;
    uint lvl;
    uint node;
    vec3 node_pos;
    float t_curr = max(grid_t_enter, 0.0f) + EPS; // the current time
    float t_max = t_curr;

    //LOG2("\nt_enter=%.3f t_leave=%.3f\n", grid_t_enter, grid_t_leave);
   
    // We use the kd-restart algorithm.
    while (t_max < grid_t_leave - EPS) {
        lvl = 0;
        node = 0;
        node_pos = params_buf.grid_world_pos;
        t_curr = t_max + EPS;
        t_max = grid_t_leave;

        while (t_curr < t_max - EPS) {
            // Prepare
            vec3 norm_pos = (orig + (t_curr + EPS) * dir - node_pos) /
                params_buf.levels[lvl].cell_size; 
            ivec3 cell = ivec3(floor(norm_pos));
            vec3 t_next = (t_curr + EPS) + (cell + SIGN_01(invDir) - norm_pos) * 
                params_buf.levels[lvl].cell_size * invDir;
            uint cell_index = INDEX3D(cell, params_buf.levels[lvl].dim);

            //LOG8("level=%u   node=%u   pos=%.3v3f   cell_size=%.4f   norm_pos=%.3v3f   cell=%v3d   cell_index=%u   node_pos=%.3v3f\n", 
            //    lvl, node, orig + (t_curr + EPS) * dir, params_buf.levels[lvl].cell_size, norm_pos, cell, cell_index, node_world_pos(node, lvl));
      
            if (node_has_leaf_child(node, lvl, cell_index)) {
                hit.t = t_curr;  
                hit.node = node;
                hit.level = lvl;
                hit.cell_index = cell_index;
                //LOG0("Hit voxel\n");
                return true;
            }
            // Recurse in the child
            else if (node_has_interior_child(node, lvl, cell_index)) {
                //LOG0("Recursing in child\n");
                node = node_get_interior_child(node, lvl, cell_index);
                node_pos += cell * params_buf.levels[lvl].cell_size;
                t_max = MIN_3(t_next.x, t_next.y, t_next.z);
                lvl++;
            }
            // No child : step forward 
            else {
                //LOG0("Stepping\n");
                t_curr = MIN_3(t_next.x, t_next.y, t_next.z);
            }
        }
    }
    // We didn't hit anything
    return false;
}



vec4 shade(vec3 ray_orig, vec3 ray_dir, Hit hit)
{
    uint tape = node_get_tape_index(hit.node, hit.level);
    vec3 normal = normalize(tape_eval_gradient(tape, ray_orig + hit.t * ray_dir));

    vec4 diffuse = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint i = 0; i < params_buf.light_count; i++) {
        DirectionalLight light = params_buf.lights[i];
        
        // Check the light illuminates the voxel
        //vec3 shadow_ray_orig = ray_orig + (hit.t - EPS) * ray_dir; 
        //Hit dummy_hit;
        //bool did_hit = dda(shadow_ray_orig, -light.direction.xyz, dummy_hit);
        //if (!did_hit) {
            // Add the light contribution

            float intensity = max(dot(-light.direction.xyz, normal), 0.0f);
            diffuse += intensity * light.color;
        //}
    }

    vec4 ambient = vec4(0.2f, 0.2f, 0.2f, 1.0f);

    return vec4(1, 1, 1, 1) * (ambient + diffuse);
}



vec4 raytrace(vec3 orig, vec3 dir)
{
    Hit hit;
    bool did_hit = dda(orig, dir, hit);
    if (did_hit) {
        // Something went wrong : show a debug color.
        if (hit.t < 0.0f) {
            return vec4(1, 1, 1, 1);
        }
        // All is good.
        return shade(orig, dir, hit);
    }
    else {
        return params_buf.background_color;
    }
}


void main()
{
    uvec2 gid = gl_GlobalInvocationID.xy;

    if (gid.x < params_buf.screen_res.x && 
        gid.y < params_buf.screen_res.y) 
    {
        random_seed(params_buf.time);

        // Compute the ray direction.
        // We add a small offset to the pixel position for antialiasing.
        vec2 ofs = 2 * vec2(rand() - 0.5f, rand() - 0.5f);
        float dx = 2.0f * ((gid.x + ofs.x) / float(params_buf.screen_res.x)) - 1.0f;
        float dy = 2.0f * ((gid.y + ofs.y) / float(params_buf.screen_res.y)) - 1.0f;
        vec3 dir = normalize(
            params_buf.camera_forward.xyz +
            dx * params_buf.screen_world_size.x * params_buf.camera_right.xyz +
            dy * params_buf.screen_world_size.y * params_buf.camera_up.xyz);

        vec4 color = raytrace(params_buf.camera_pos.xyz, dir);
        ivec3 coords = ivec3(gid, params_buf.out_img_layer);
        imageStore(out_img, coords, color);
    }
}
