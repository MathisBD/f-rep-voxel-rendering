#include "engine/camera.h"
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <assert.h>
#include <stdio.h>
#include "engine/timer.h"


void Camera::Init() 
{
    forward = glm::normalize(forward);
    initialUp = glm::normalize(initialUp);
    m_rotating = false;
}

glm::vec3 Camera::Right() const
{
    // We normalize because the forward and initial up vectors
    // are not necessarily orthogonal.
    return glm::normalize(glm::cross(forward, initialUp));       
}

glm::vec3 Camera::Up() const 
{
    // We return the CURRENT up vector (not the initial one).
    return glm::cross(Right(), forward);    
}


void Camera::Move(float x, float y, float z) 
{
    position += Timer::s_dt * moveSpeed * x * Right();
    position += Timer::s_dt * moveSpeed * y * initialUp;
    position += Timer::s_dt * moveSpeed * z * forward;
}

void Camera::RotateHorizontal(float x) 
{
    float angle = glm::radians(Timer::s_dt * rotateSpeed * x);
    forward = glm::rotate(forward, angle, initialUp);    
}

void Camera::RotateVertical(float y) 
{
    float angle = glm::radians(Timer::s_dt * rotateSpeed * y);
    glm::vec3 newForward = glm::rotate(forward, angle, Right());  
    if (glm::angle(newForward, initialUp) > glm::radians(10.0f) &&
        glm::angle(newForward, -initialUp) > glm::radians(10.0f)) {
        forward = newForward;
    }  
}

bool Camera::Update(const InputManager& input) 
{
    bool changed = false;
    // translate camera
    if (input.keyState.left == InputManager::KState::PRESSED) {
        Move(-1.0, 0.0, 0.0);
        changed = true;
    }
    if (input.keyState.right == InputManager::KState::PRESSED) {
        Move(1.0, 0.0, 0.0);
        changed = true;
    }
    if (input.keyState.down == InputManager::KState::PRESSED) {
        Move(0.0, 0.0, -1.0);
        changed = true;
    }
    if (input.keyState.up == InputManager::KState::PRESSED) {
        Move(0.0, 0.0, 1.0);
        changed = true;
    }
    if (input.keyState.rShift == InputManager::KState::PRESSED) {
        Move(0.0, 1.0, 0.0);
        changed = true;
    }
    if (input.keyState.rCtrl == InputManager::KState::PRESSED) {
        Move(0.0, -1.0, 0.0);
        changed = true;
    }

    // rotate camera
    if (input.keyState.mouseRight == InputManager::KState::PRESSED) {
        if (!m_rotating) {
            //input->DisableCursor();
            m_prevCursorPos = input.CursorPosition();
        }
        m_rotating = true;
    }
    if (input.keyState.mouseRight == InputManager::KState::RELEASED) {
        if (m_rotating) {
            //input->ShowCursor();
        }
        m_rotating = false;
    }
    if (m_rotating) {
        glm::vec2 cursorPos = input.CursorPosition();
        RotateHorizontal(-(cursorPos.x - m_prevCursorPos.x));
        RotateVertical(cursorPos.y - m_prevCursorPos.y);
        m_prevCursorPos = cursorPos;
        changed = true;
    }
    return changed;
}
