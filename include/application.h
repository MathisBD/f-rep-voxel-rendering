#pragma once
#include "engine/engine_base.h"
#include "vk_wrapper/image.h"
#include "vk_wrapper/buffer.h"
#include <vector>
#include "engine/render_target.h"
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "engine/camera.h"
#include "engine/cube_grid.h"
#include "engine/renderer.h"


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

    RenderTarget m_target;
    Renderer m_renderer;


    void InitRenderTarget();

    void UpdateDDAVoxels();
    void UpdateDDAUniforms();
    void UpdateDDALights();
    
    VkCommandBuffer BuildCommand(
        VkCommandPool pool, 
        std::function<void(VkCommandBuffer)>&& record);
    void Draw() override;
};