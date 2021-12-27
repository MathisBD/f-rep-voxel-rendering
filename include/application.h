#pragma once
#include "engine/engine_base.h"
#include "vk_wrapper/image.h"
#include <vector>
#include "engine/render_target.h"
#include "glm/glm.hpp"
#include "engine/camera.h"
#include "engine/renderer.h"
#include "engine/raytracer.h"
#include "utils/running_average.h"
#include "engine/scene_builder.h"
#include "engine/voxel_storage.h"


class Application : public EngineBase
{
public:
    Application();
    virtual void Init(bool enableValidationLayers) override;
private:
    VoxelStorage m_voxels;
    RenderTarget m_target;
    Camera m_camera;

    SceneBuilder m_builder;
    Raytracer m_raytracer;
    Renderer m_renderer;
    
    RunningAverage<float> m_frameTime;

    void InitVoxels();
    void InitRenderTarget();
    void Draw() override;
};