#include "engine/render_target.h"
#include "vk_wrapper/initializers.h"
#include "vk_wrapper/vk_check.h"
#include <cstring>


void RenderTarget::Init(vkw::Device* device, VkExtent2D windowExtent, uint32_t samples) 
{
    m_device = device;
    temporalSampleCount = samples;
    m_windowExtent = windowExtent;
    
    image.Init(m_device->vmaAllocator);
    AllocateImage();
    InitView();
    InitSampler();
}

void RenderTarget::Cleanup() 
{
    m_cleanupQueue.Flush();    
}

void RenderTarget::AllocateImage() 
{
    std::vector<uint32_t> queueFamilies = {
            m_device->queueFamilies.graphics,
            m_device->queueFamilies.compute,
            m_device->queueFamilies.transfer };
    VkSharingMode sharingMode = VK_SHARING_MODE_CONCURRENT;

    image.Allocate(
        m_windowExtent,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY,
        sharingMode,
        &queueFamilies);
    m_cleanupQueue.AddFunction([=] { image.Cleanup(); });
}

void RenderTarget::ZeroOutImage(VkCommandBuffer cmd) 
{
    // Create the staging buffer
    m_stagingBuf.Init(m_device);
    m_stagingBuf.Allocate(m_windowExtent.width * m_windowExtent.height * temporalSampleCount,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    m_cleanupQueue.AddFunction([=] { m_stagingBuf.Cleanup(); });

    // Zero its contents
    void* contents = m_stagingBuf.Map();
    memset(contents, 0, m_stagingBuf.size);
    m_stagingBuf.Unmap();

    // Change the layout to TRANSFER_DST
    image.ChangeLayout(cmd, 
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, VK_ACCESS_TRANSFER_WRITE_BIT);

    image.CopyFromBuffer(cmd, &m_stagingBuf);

    // Change the layout to GENERAL
    image.ChangeLayout(cmd,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
}

void RenderTarget::InitView() 
{
    auto info = vkw::init::ImageViewCreateInfo(
        image.format, image.image, VK_IMAGE_ASPECT_COLOR_BIT);
    info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    info.subresourceRange.layerCount = temporalSampleCount;

    VK_CHECK(vkCreateImageView(m_device->logicalDevice, &info, nullptr, &view));
    m_cleanupQueue.AddFunction([=] {
        vkDestroyImageView(m_device->logicalDevice, view, nullptr);
    });
}

void RenderTarget::InitSampler() 
{
    auto samplerInfo = vkw::init::SamplerCreateInfo(VK_FILTER_NEAREST);

    VK_CHECK(vkCreateSampler(
        m_device->logicalDevice, &samplerInfo, nullptr, &sampler));
    m_cleanupQueue.AddFunction([=] {
        vkDestroySampler(m_device->logicalDevice, sampler, nullptr);
    });
}