#pragma once
#include <stdio.h>
#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <assert.h>


#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = (x);                                         \
		if (err)                                                    \
		{                                                           \
			printf("[-] Vulkan error: %d\n\tfile: %s:%d\n",          \
				err, __FILE__, __LINE__);                           \
			assert(false);                                               \
		}                                                           \
	} while (0)
	