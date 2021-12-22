#pragma once
#include "vk_wrapper/image.h"


// Essentially an image, drawed to by the raytracer
// and used by the renderer to create a frame.
struct RenderTarget
{
    vkw::Image image;
    VkImageView view;
    VkSampler sampler;
};