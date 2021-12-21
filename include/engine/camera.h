#pragma once
#include <glm/glm.hpp>
#include "engine/input_manager.h"


class Camera
{
public:
    // The user has to fill these parameters manually, then call Init();
    glm::vec3 position;
    glm::vec3 forward;
    // The initial up direction stays constant, 
    // and is used when moving the camera.
    glm::vec3 initialUp;
    float fovDeg; // horizontal field of view in degrees
    float aspectRatio; // screen width / screen height
    float moveSpeed = 20.0f; // units per second 
    float rotateSpeed = 2000.0f;

    void Init();
        
    // Returns true if the camera's state (position/rotation)
    // has changed.
    bool Update(const InputManager& input);
    
    // The forward, right and up vectors are normalized and orthogonal.
    glm::vec3 Right() const;
    // Returns the current (not initial) up axis
    glm::vec3 Up() const;
private:
    bool m_rotating;
    glm::vec2 m_prevCursorPos;

    // Translate the camera : 
    // x is the local left/right axis
    // y is the INITIAL up axis.
    // z is the local forward/backwards axis
    void Move(float x, float y, float z);
    void RotateHorizontal(float x);
    void RotateVertical(float y);

    void UpdateMatrix();
    void UpdateFrustrumPlanes();
};