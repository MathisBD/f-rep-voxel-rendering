// this definition should go in exactly one .cpp file
// that includes the VMA header file.
#define VMA_IMPLEMENTATION

// The VMA implementation compiles with warnings : disable them temporarily.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wreorder"
#include "third_party/vk_mem_alloc.h"
#pragma GCC diagnostic pop