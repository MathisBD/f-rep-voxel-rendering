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


class Application : public EngineBase
{
public:
    Application();
    virtual void Init(bool enableValidationLayers) override;
private:
    RenderTarget m_target;
    Renderer m_renderer;
    Raytracer m_raytracer;
    Camera m_camera;

    RunningAverage<float> m_frameTime;

    void InitRenderTarget();
    void Draw() override;
};