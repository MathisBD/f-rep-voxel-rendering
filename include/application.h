#pragma once
#include "engine/engine_base.h"
#include "vk_wrapper/image.h"
#include <vector>
#include "engine/render_target.h"
#include "glm/glm.hpp"
#include "engine/camera.h"
#include "engine/renderer.h"
#include "engine/raytracer.h"
#include "engine/voxelizer.h"
#include "utils/running_average.h"
#include "engine/scene_builder.h"
#include "engine/voxel_storage.h"


class Application : public EngineBase
{
public:
    struct Params
    {
        bool enableValidationLayers = true;
        bool enableShaderDebugPrintf = false;
        bool voxelizeRealTime = true;
        bool printFPS = false;
        bool printHardwareInfo = false;
        std::vector<uint32_t> gridDims;
        csg::Expr shape;
        uint32_t temporalSampleCount = 1;
    };

    Application(Params params);
    virtual void Init() override;
private:
    Params m_params;
    VoxelStorage m_voxels;
    RenderTarget m_target;
    Camera m_camera;

    SceneBuilder m_builder;
    Voxelizer m_voxelizer;
    Raytracer m_raytracer;
    Renderer m_renderer;
    
    RunningAverage<float> m_frameTime;
    bool m_voxelizedOnce = false;

    void Draw() override;
    void PrintInfo();
};