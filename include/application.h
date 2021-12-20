#pragma once
#include "engine/engine_base.h"
#include "vk_wrapper/image.h"
#include <vector>
#include "engine/swapchain.h"


class Application : public EngineBase
{
public:
    virtual void Init() override;
private:
    vkw::Image m_image;
    VkImageView m_imageView;
    VkSampler m_sampler;
    
    struct {
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
        std::vector<VkDescriptorSet> dSets;
    
        VkQueue queue;
        VkCommandPool cmdPool;
        // signaled when the graphics command is finished,
        // waited on by the compute command.
        VkSemaphore semaphore;
        // signaled when the graphics command is finished,
        // waited on by the present command;
        VkSemaphore presentSem;
        // signaled when the swapchain image is acquired
        VkSemaphore imageReadySem;
        
        Swapchain swapchain;
        // signaled when the graphics command is finished
        VkFence fence;
    } m_graphics;

    struct {
        VkPipeline pipeline;
        VkPipelineLayout pipelineLayout;
        std::vector<VkDescriptorSet> dSets;
        
        VkQueue queue;
        VkCommandPool cmdPool;
        // signaled when the compute command is finished
        VkSemaphore semaphore;
        // signaled when the compute command is finished
        VkFence fence; 
    } m_compute;

    void InitImage();

    void InitGraphicsPipeline();
    void InitGraphics();
    void RecordGraphicsCmd(VkCommandBuffer cmd, uint32_t swapchainImgIdx);
    void SubmitGraphicsCmd(VkCommandBuffer cmd);

    void InitComputePipeline();
    void InitCompute();
    void RecordComputeCmd(VkCommandBuffer cmd);
    void SubmitComputeCmd(VkCommandBuffer cmd);

    VkCommandBuffer BuildCommand(
        VkCommandPool pool, 
        std::function<void(VkCommandBuffer)>&& record);
    void Draw() override;
};