#version 450

#extension GL_EXT_debug_printf : enable

// Include other files  
#define DEBUG_LOG 
const float infinity = 1. / 0.;
const float pi = 3.141592653589793238462643383279502884197;

#define CUBE(x) ((x)*(x)*(x))

#define INDEX3D(cell, dim) ((cell).z + (dim) * (cell).y + (dim) * (dim) * (cell).x)

#define IS_BIT_SET(x, bit) ((x) & (1 << (bit)))

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


#ifdef DEBUG_LOG

#define SHOULD_LOG() (gl_GlobalInvocationID.xyz == vec3(0, 2, 2))

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

#define THREAD_GROUP_SIZE_X 4
#define THREAD_GROUP_SIZE_Y 4
#define THREAD_GROUP_SIZE_Z 4
// The valid levels range from 0 to LEVEL_COUNT-1 included.
// Level 0 contains the unique root node, and LEVEL_COUNT-1 the last interior nodes,
// whose only children are leafs (and empty nodes).
#define LEVEL_COUNT 4
// The maximum number of tape slots.
#define MAX_SLOT_COUNT 256
// The maximum number of instructions per tape.
#define MAX_TAPE_SIZE 4096


#define MAX_LEVEL_COUNT 8
#define MAX_CONST_POOL_SIZE 256

// Each node of the previous level is handled by
// several thread groups.
layout (
    local_size_x = THREAD_GROUP_SIZE_X, 
    local_size_y = THREAD_GROUP_SIZE_Y, 
    local_size_z = THREAD_GROUP_SIZE_Z) in;

struct LevelData {
    uint dim;
    // These are offsets in the node and child buffers.
    // They are given in uints rather than in bytes.
    uint node_ofs;
    // The world size of a single child cell at this level.
    // The cell_size of the last level (level_count-1) is the voxel size.
    // The cell_size of level 0 is world_size / dim[0].
    float cell_size;
    uint _padding_;
};


// Shader basic data.
layout (set = 0, binding = 0) uniform ParamsBuffer {
    uint level;
    float tape_time; // the time value used to evaluate the tape.
    uint _padding_0;
    uint _padding_1;

    vec3 grid_world_pos;
    float grid_world_size;

    LevelData levels[MAX_LEVEL_COUNT];
    // The tape constant pool.
    vec4 const_pool[MAX_CONST_POOL_SIZE / 4];
} params_buf;

// Nodes
// We need the std430 layout here so that the array stride is 4 bytes.
// The std140 layout would round it up to 16 bytes.
layout (std430, set = 0, binding = 1) buffer NodeBuffer {
    uint data[];
} node_buf;

layout (std430, set = 0, binding = 2) buffer TapeBuffer {
    uint data[];
} tape_buf;

// This buffer contains counters that are atomically incremented
// by threads.
layout (std430, set = 0, binding = 3) buffer CountersBuffer {
    uint child_count;
    uint tape_index;
} counters_buf;

layout (std430, set = 0, binding = 4) buffer StatsBuffer {
    uint tape_size_sum[MAX_LEVEL_COUNT];
    uint max_tape_size[MAX_LEVEL_COUNT];
} stats_buf;


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


float tape_eval_point(uint tape, vec3 pos)
{
    float slots[MAX_SLOT_COUNT];
    slots[0] = pos.x;
    slots[1] = pos.y;
    slots[2] = pos.z;
    slots[3] = params_buf.tape_time;

    uint idx = 0;
    uint size = tape_read_size(tape);
    while (idx < size) {
        // Decode the instruction
        Instruction i = tape_read_inst(tape, idx);
        float a = slots[i.inSlotA];
        float b = slots[i.inSlotB];

        // Dispatch        
        switch (i.op) {
        case OP_CONSTANT:
            slots[i.outSlot] = params_buf.const_pool[i.inSlotA / 4][i.inSlotA % 4];
            break;
        case OP_SIN:
            slots[i.outSlot] = sin(a);
            break;
        case OP_COS:
            slots[i.outSlot] = cos(a);       
            break;
        case OP_ADD:
            slots[i.outSlot] = a + b;
            break;
        case OP_SUB:
            slots[i.outSlot] = a - b;
            break;
        case OP_MUL:
            slots[i.outSlot] = a * b;
            break;
        case OP_DIV:
            slots[i.outSlot] = a / b;
            break;
        case OP_MIN:
            slots[i.outSlot] = min(a, b);       
            break;
        case OP_MAX:
            slots[i.outSlot] = max(a, b);       
            break;
        case OP_EXP:
            slots[i.outSlot] = exp(a);       
            break;
        case OP_NEG:
            slots[i.outSlot] = -a;       
            break;
        case OP_SQRT:
            slots[i.outSlot] = sqrt(a);       
            break;
        case OP_COPY:
            slots[i.outSlot] = a;       
            break;
        } 
        idx++;
    }   
    return slots[0];
}

struct Interval {
    float low;
    float high;
};


Interval interval_multiply(Interval a, Interval b)
{
    Interval c;
    c.low = MIN_4(
        a.low * b.low, a.low * b.high, a.high * b.low, a.high * b.high);
    c.high = MAX_4(
        a.low * b.low, a.low * b.high, a.high * b.low, a.high * b.high);
    return c;
}

// Returns true if an integral multiple of x lies
// in the interval i (endpoints included).
bool contains_multiple(Interval i, float x)
{
    return floor(i.low / x) < floor(i.high / x) || 
           floor(i.low / x) == i.low / x        ||
           floor(i.high / x) == i.high / x;
}


// First erase the old choice, then OR with the new choice.
#define MM_STORE(mm_array, idx, choice) { \
    uint q = (2*(idx)) >> 5;              \
    uint r = (2*(idx)) & 0x1F;            \
    mm_array[q] &= ~(0x3 << r);           \
    mm_array[q] |= (choice) << r;         \
}

#define INTERVAL_UPDATE_SLOTS()\
switch (i.op) {\
case OP_CONSTANT:\
    float c = params_buf.const_pool[i.inSlotA / 4][i.inSlotA % 4];\
    slots[i.outSlot] = Interval(c, c);\
    break;\
case OP_COS:\
    if (contains_multiple(Interval(a.low - pi, a.high - pi), 2*pi)) {\
        slots[i.outSlot].low = -1;\
    }\
    else {\
        slots[i.outSlot].low = min(cos(a.low), cos(a.high));\
    }\
    if (contains_multiple(a, 2*pi)) {\
        slots[i.outSlot].high = 1;\
    }\
    else {\
        slots[i.outSlot].high = max(cos(a.low), cos(a.high));\
    }\
    break;\
case OP_SIN:\
    if (contains_multiple(Interval(a.low + pi / 2, a.high + pi / 2), 2*pi)) {\
        slots[i.outSlot].low = -1;\
    }\
    else {\
        slots[i.outSlot].low = min(sin(a.low), sin(a.high));\
    }\
    if (contains_multiple(Interval(a.low - pi / 2, a.high - pi / 2), 2*pi)) {\
        slots[i.outSlot].high = 1;\
    }\
    else {\
        slots[i.outSlot].high = max(sin(a.low), sin(a.high));\
    }\
    break;\
case OP_ADD:\
    slots[i.outSlot] = Interval(a.low + b.low, a.high + b.high);\
    break;\
case OP_SUB:\
    slots[i.outSlot] = Interval(a.low - b.high, a.high - b.low);\
    break;\
case OP_MUL:\
    slots[i.outSlot] = interval_multiply(a, b);\
    break;\
case OP_DIV:\
    if ((b.low == 0 && b.high == 0) || (b.low < 0 && b.high > 0)) {\
        slots[i.outSlot] = Interval(-infinity, infinity);\
    }\
    else if (b.low == 0) {\
        slots[i.outSlot] = interval_multiply(a, Interval(1. / b.high, infinity));\
    }\
    else if (b.high == 0) {\
        slots[i.outSlot] = interval_multiply(a, Interval(-infinity, 1. / b.low));\
    }\
    else {\
        slots[i.outSlot] = interval_multiply(a, Interval(1. / b.high, 1. / b.low));\
    }\
    break;\
case OP_MIN: \
    slots[i.outSlot] = Interval(min(a.low, b.low), min(a.high, b.high));\
    break;\
case OP_MAX:\
    slots[i.outSlot] = Interval(max(a.low, b.low), max(a.high, b.high));\
    break;\
case OP_EXP:\
    slots[i.outSlot] = Interval(exp(a.low), exp(a.high));\
    break;\
case OP_NEG:\
    slots[i.outSlot] = Interval(-a.high, -a.low);\
    break;\
case OP_SQRT:\
    slots[i.outSlot] = Interval(sqrt(max(0, a.low)), sqrt(max(0, a.high)));\
    break;\
case OP_COPY:\
    slots[i.outSlot] = a;\
    break;\
}

// How to compute intervals for unary C^1 functions. 
// If the input interval is [x, y] and we call [f(a), f(b)] the output interval,
// then it must hold that : 
// (i) f'(a) == 0 or a == x or a == y
// (ii) f'(b) == 0 or b == x or b == y
// Thus, if c_1 ... c_n are the critical points of f in [x, y] (i.e. where f' == 0),
// the output interval is [min(f(x), f(y), f(c_i)), max(f(x), f(y), f(c_i))].
// We use this method for sin and cos (for which we know the critical points).
Interval tape_eval_interval(uint tape, vec3 low, vec3 high, 
    out uint mm_array[MM_ARRAY_SIZE], out uint mm_size)
{
    Interval slots[MAX_SLOT_COUNT];
    slots[0] = Interval(low.x, high.x);
    slots[1] = Interval(low.y, high.y);
    slots[2] = Interval(low.z, high.z);
    slots[3] = Interval(params_buf.tape_time, params_buf.tape_time);

    uint idx = 0;
    uint size = tape_read_size(tape);
    uint mm_idx = 0;
    while (idx < size) {
        // Decode the instruction
        Instruction i = tape_read_inst(tape, idx);
        Interval a = slots[i.inSlotA];
        Interval b = slots[i.inSlotB];

        INTERVAL_UPDATE_SLOTS();
        if (i.op == OP_MIN) {
            uint choice = 
                a.high < b.low ? MM_FIRST  :
                b.high < a.low ? MM_SECOND :
                MM_BOTH;
            MM_STORE(mm_array, mm_idx, choice);
            mm_idx++;
        }
        else if (i.op == OP_MAX) {
            uint choice = 
                a.high < b.low ? MM_SECOND :
                b.high < a.low ? MM_FIRST  :
                MM_BOTH;
            MM_STORE(mm_array, mm_idx, choice);
            mm_idx++;
        }
        idx++;
    }   
    mm_size = mm_idx;
    return slots[0];
}

// No tape shortening.
Interval tape_eval_interval(uint tape, vec3 low, vec3 high)
{
    Interval slots[MAX_SLOT_COUNT];
    slots[0] = Interval(low.x, high.x);
    slots[1] = Interval(low.y, high.y);
    slots[2] = Interval(low.z, high.z);
    slots[3] = Interval(params_buf.tape_time, params_buf.tape_time);

    uint idx = 0;
    uint size = tape_read_size(tape);
    while (idx < size) {
        // Decode the instruction
        Instruction i = tape_read_inst(tape, idx);
        Interval a = slots[i.inSlotA];
        Interval b = slots[i.inSlotB];
        INTERVAL_UPDATE_SLOTS();
        idx++;
    }   
    return slots[0];
}


void tape_write_size(uint tape_idx, uint size)
{
    tape_buf.data[tape_idx] = size;
}

void tape_write_inst(uint tape, uint idx, Instruction inst)
{
    uint data = 
        ((inst.op      & 0xFF) << 0)  |
        ((inst.outSlot & 0xFF) << 8)  |
        ((inst.inSlotA & 0xFF) << 16) |
        ((inst.inSlotB & 0xFF) << 24);
    tape_buf.data[tape + idx + 1] = data;
}


uint mm_load(uint mm_array[MM_ARRAY_SIZE], uint idx)
{
    uint q = (2*idx) >> 5;
    uint r = (2*idx) & 0x1F;
    return (mm_array[q] >> r) & 0x3;
}

uint get_choice(uint op, uint mm_array[MM_ARRAY_SIZE], uint mm_idx)
{
    uint choice;
    switch (op) {
    case OP_CONSTANT: 
        choice = MM_NONE; break;
    case OP_COS:
    case OP_SIN:
    case OP_EXP:
    case OP_NEG:
    case OP_SQRT:
    case OP_COPY:
        choice = MM_FIRST; break;
    case OP_ADD:
    case OP_SUB:
    case OP_MUL:
    case OP_DIV:
        choice = MM_BOTH; break;
    case OP_MIN:
    case OP_MAX:
        choice = mm_load(mm_array, mm_idx);
        break;
    } 
    return choice;
}


// Returns the tape index of the output tape
// (this can be equal to the input tape if there is nothing to shorten).
uint tape_shorten(uint in_tape, 
    uint mm_array[MM_ARRAY_SIZE], uint mm_size)
{
    LOG0("[+] Tape shortening\n");
    LOG1("mm_array[0]=0x%x\n", mm_array[0]);
    // The i-th bit is set if the i-th slot is active.
    // Initially only the output slot is active.
    uint slots[MAX_SLOT_COUNT / 32];
    for (uint i = 0; i < (MAX_SLOT_COUNT / 32); i++) {
        slots[i] = 0;
    }
    BITSET_SET(slots, 0, 0);
    
    // The i-th bit is set if the i-th instruction is kept in the shortened tape.
    uint keep[MAX_TAPE_SIZE / 32];
    for (uint i = 0; i < (MAX_TAPE_SIZE / 32); i++) {
        keep[i] = 0;
    }
    // This counts the number of kept instructions.
    uint keepCount = 0;

    // First iterate in reverse over the tape and 
    // record which instructions to keep.
    int idx = int(tape_read_size(in_tape) - 1);
    uint mm_idx = mm_size - 1;
    while (idx >= 0) {
        // Decode the instruction
        Instruction i = tape_read_inst(in_tape, idx);
        
        LOG5("\t%d: op=%u  out=%u  inA=%u  inB=%u\n", 
            idx, i.op, i.outSlot, i.inSlotA, i.inSlotB);

        if (BITSET_LOAD(slots, 0, i.outSlot)) {
            LOG0("\t\tkeep\n");
            // We will keep this instruction.
            BITSET_SET(keep, 0, idx);
            keepCount++;
            // Activate the right slots.
            uint choice = get_choice(i.op, mm_array, mm_idx);
            LOG2("\t\tchoice=%u  mm_idx=%u\n", choice, mm_idx);
            BITSET_CLEAR(slots, 0, i.outSlot);
            if (bool(choice & MM_FIRST))  { BITSET_SET(slots, 0, i.inSlotA); }
            if (bool(choice & MM_SECOND)) { BITSET_SET(slots, 0, i.inSlotB); }
        }
        if (i.op == OP_MIN || i.op == OP_MAX) { mm_idx--; }    
        idx--;
    }   

    LOG1("keep count=%u\n", keepCount);

    // The shortened tape is not much shorter :
    // don't copy it.
    uint in_size = tape_read_size(in_tape);
    if (keepCount >= in_size) {
        return in_tape;
    }

    // Claim space for the output tape.
    uint out_tape = atomicAdd(counters_buf.tape_index, 1 + keepCount);
    tape_write_size(out_tape, keepCount);

    // Then forward iterate over the tape and copy the relevant instructions.
    idx = 0;
    mm_idx = 0;
    uint out_idx = 0;
    while (idx < in_size) {
        // Decode the input instruction.
        Instruction i = tape_read_inst(in_tape, idx);
        
        LOG5("\t%d: op=%u  out=%u  inA=%u  inB=%u\n", 
            idx, i.op, i.outSlot, i.inSlotA, i.inSlotB);
        
        // We have to store this as i.op will be overwritten.
        bool is_mm_op = (i.op == OP_MIN || i.op == OP_MAX);
        if (BITSET_LOAD(keep, 0, idx)) {
            LOG0("\t\tkeep\n");
            // Change the min/max operation to a copy operation.
            if (i.op == OP_MIN || i.op == OP_MAX) {
                uint choice = mm_load(mm_array, mm_idx);
                LOG2("\t\tchoice=%u  mm_idx=%u\n", choice, mm_idx);
                // A copy operation only uses its first input (A).
                if      (choice == MM_FIRST)  { i.op = OP_COPY; }
                else if (choice == MM_SECOND) { i.op = OP_COPY; i.inSlotA = i.inSlotB; }
            }
            // Copy the instruction.
            tape_write_inst(out_tape, out_idx, i);
            out_idx++;
        }
        if (is_mm_op) { mm_idx++; }
        idx++;
    }
    return out_tape;
}



#define LVL params_buf.level
#define DIM params_buf.levels[params_buf.level].dim

void set_leaf_bit(uint node, uvec3 cell, bool set)
{
    uint cell_index = INDEX3D(cell, DIM);
    uint mask_pos = params_buf.levels[LVL].node_ofs +
        node * NODE_SIZE(LVL) + NODE_OFS_LEAF_MASK(LVL);     

    if (set) { 
        BITSET_ATOMIC_SET(node_buf.data, mask_pos, cell_index); 
    }
    else { 
        BITSET_ATOMIC_CLEAR(node_buf.data, mask_pos, cell_index); 
    }
}

void set_interior_bit(uint node, uvec3 cell, bool set)
{
    uint cell_index = INDEX3D(cell, DIM);
    uint mask_pos = params_buf.levels[LVL].node_ofs +
        node * NODE_SIZE(LVL) + NODE_OFS_INTERIOR_MASK(LVL);     

    if (set) { 
        BITSET_ATOMIC_SET(node_buf.data, mask_pos, cell_index); 
    }
    else { 
        BITSET_ATOMIC_CLEAR(node_buf.data, mask_pos, cell_index); 
    }
}

void set_child_list_entry(uint node, uvec3 cell, uint child)
{
    // set the child id in the parent child list
    uint child_list_pos = params_buf.levels[LVL].node_ofs +
        node * NODE_SIZE(LVL) +
        NODE_OFS_CHILD_LIST(LVL);

    uint cell_index = INDEX3D(cell, DIM);
    node_buf.data[child_list_pos + cell_index] = child;
}

void set_child_coords(uint child, uvec3 coords)
{
    uint coords_pos = params_buf.levels[LVL+1].node_ofs +
        child * NODE_SIZE(LVL+1) + NODE_OFS_COORDS(LVL+1);
    node_buf.data[coords_pos + 0] = coords.x;
    node_buf.data[coords_pos + 1] = coords.y;
    node_buf.data[coords_pos + 2] = coords.z;
}

void set_child_tape_idx(uint child, uint tape_idx)
{
    uint idx_pos = params_buf.levels[LVL+1].node_ofs +
        child * NODE_SIZE(LVL+1) + NODE_OFS_TAPE_IDX(LVL+1);
    node_buf.data[idx_pos] = tape_idx;
}

void voxelize_point(uint node, uvec3 cell)
{
    uint tape = node_get_tape_index(node, LVL);
    uvec3 child_coords = node_get_coords(node, LVL) * DIM + cell;
    vec3 child_pos = params_buf.grid_world_pos + 
        child_coords * params_buf.levels[LVL].cell_size;
    
    float density = tape_eval_point(tape, child_pos);
    set_leaf_bit(node, cell, density < 0.0f);
}

// Returns true if the child node is ambiguous,
// i.e. interval evalution didn't classify it as 
// a leaf or empty node.
bool set_mask_bits(Interval density, uint node, uvec3 cell)
{
    // Empty node
    if (density.low > 0) {
        set_leaf_bit(node, cell, false);
        set_interior_bit(node, cell, false);
        return false;
    }  
    // Leaf node
    else if (density.high < 0) {
        set_leaf_bit(node, cell, true);
        set_interior_bit(node, cell, false);
        return false;
    }
    // Ambiguous node    
    else {
        set_leaf_bit(node, cell, false);
        set_interior_bit(node, cell, true);
        return true;
    }
}


void voxelize_interval(uint node, uvec3 cell, bool shorten)
{
    uint tape = node_get_tape_index(node, LVL);
    uvec3 child_coords = node_get_coords(node, LVL) * DIM + cell;
    vec3 child_pos = params_buf.grid_world_pos + 
        child_coords * params_buf.levels[LVL].cell_size;
    
    bool ambiguous;
    uint child;
    uint child_tape;
    if (shorten) {
        uint mm_array[MM_ARRAY_SIZE];
        uint mm_size;
        Interval density = tape_eval_interval(tape, 
            child_pos, child_pos + params_buf.levels[LVL].cell_size,
            mm_array, mm_size);
        ambiguous = set_mask_bits(density, node, cell);
        if (ambiguous) {
            child = atomicAdd(counters_buf.child_count, 1);
            child_tape = tape_shorten(tape, mm_array, mm_size);
        }
    }
    else {
        Interval density = tape_eval_interval(tape, 
            child_pos, child_pos + params_buf.levels[LVL].cell_size);
        ambiguous = set_mask_bits(density, node, cell);
        if (ambiguous) {
            child = atomicAdd(counters_buf.child_count, 1);
            child_tape = tape;
        }
    }  

    if (ambiguous) {
        // Setup the child node
        set_child_list_entry(node, cell, child);
        set_child_coords(child, child_coords);
        set_child_tape_idx(child, child_tape);

        // Collect some stats
        uint child_size = tape_read_size(child_tape);
        atomicAdd(stats_buf.tape_size_sum[LVL+1], child_size);
        atomicMax(stats_buf.max_tape_size[LVL+1], child_size);
    }
}


// The first stage only updates the leaf and interior masks.
// All the other book-keeping operations are performed in the second stage. 
void main()
{ 
    uvec3 cell = gl_GlobalInvocationID.xyz % DIM;
    uint node = gl_GlobalInvocationID.x / DIM;

    if (LVL == LEVEL_COUNT - 1) {
        voxelize_point(node, cell);
    }
    else {
        //bool shorten = LVL < (LEVEL_COUNT - 2);
        bool shorten = true;
        voxelize_interval(node, cell, shorten);
    }
} 
