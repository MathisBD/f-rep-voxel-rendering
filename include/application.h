#pragma once
#include "engine/engine_base.h"
#include "vk_wrapper/image.h"
#include <vector>


class Application : public EngineBase
{
public:
    virtual void Init() override;
private:
    typedef struct {
        glm::vec4 color;
    } GPUCameraData;

    vkw::Buffer m_cameraBuffer;
    vkw::Image m_image;
    VkPipeline m_pipeline; 
    VkPipelineLayout m_pipelineLayout;
    std::vector<VkDescriptorSet> m_dSets;

    void InitBuffer();
    void InitImage();
    void InitPipelines();

    void Draw() override;
};