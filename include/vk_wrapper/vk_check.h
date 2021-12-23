#pragma once
#include <stdio.h>
#include <vulkan/vulkan.h>
#include <stdlib.h>


#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = (x);                                         \
		if (err)                                                    \
		{                                                           \
			printf("[-] Vulkan error: %d\n\tfile: %s:%d\n",          \
				err, __FILE__, __LINE__);                           \
			exit(-1);                                               \
		}                                                           \
	} while (0)
	