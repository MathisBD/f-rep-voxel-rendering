#pragma once
#include "engine/engine_base.h"
#include "vk_wrapper/image.h"
#include "vk_wrapper/buffer.h"
#include <vector>
#include "engine/swapchain.h"
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "engine/camera.h"


class Application : public EngineBase
{
public:
    virtual void Init() override;
private:
    typedef struct {     
        // The screen resolution in pixels (zw unused).
        glm::uvec4 screenResolution;
        // The world size of the screen boundaries
        // at one unit away from the camera position (zw unused).
        glm::vec4 screenWorldSize;

        // The camera world position (w unused).
        glm::vec4 cameraPosition;
        // The direction the camera is looking in (w unused).
        // The camera forward, up and right vectors are normalized
        // and orthogonal.
        glm::vec4 cameraForward;
        glm::vec4 cameraUp;
        glm::vec4 cameraRight;

        // The world positions of the grid bottom left corner (xyz)
        // and the world size of the grid (w).
        glm::vec4 gridWorldCoords;
        // The number of subdivisions along each grid axis (w unused).
        glm::uvec4 gridResolution;
    } DDAUniforms;

    typedef struct {
        glm::vec4 color;
        // The normal vector at the center of the voxel (w unused).
        glm::vec4 normal;
    } DDAVoxel;

    // A single point light.
    typedef struct {
        glm::vec4 color;
        // The direction the light is pointing (w unused).
        glm::vec4 direction;
    } DDALight;

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

        vkw::Buffer ddaUniforms;
        vkw::Buffer ddaVoxels;
        const size_t gridResolution = 128;
        vkw::Buffer ddaLights;
        const size_t lightCount = 2;
        Camera camera;
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

    void UpdateDDAVoxels();
    void UpdateDDAUniforms();
    void UpdateDDALights();
    VkCommandBuffer BuildCommand(
        VkCommandPool pool, 
        std::function<void(VkCommandBuffer)>&& record);
    void Draw() override;
};