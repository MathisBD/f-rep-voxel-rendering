#pragma once
#include <iostream>
#include <vulkan/vulkan.h>


#define VK_CHECK(x)                                                 \
	do                                                              \
	{                                                               \
		VkResult err = (x);                                           \
		if (err)                                                    \
		{                                                           \
			std::cout <<"Detected Vulkan error: " << err << std::endl; \
			exit(-1);                                                \
		}                                                           \
	} while (0)
	