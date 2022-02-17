#include "engine/input_manager.h"
#include <assert.h>


void InputManager::Init(const glm::uvec2& windowPixelSize) 
{
    m_windowPixelSize = windowPixelSize;
}

void InputManager::UpdateKey(const SDL_Event& e)
{ 
    if (e.type == SDL_KEYDOWN) {
        switch (e.key.keysym.sym) {
        case SDLK_LEFT: keyState.left = KState::PRESSED; break;
        case SDLK_RIGHT: keyState.right = KState::PRESSED; break;
        case SDLK_UP: keyState.up = KState::PRESSED; break;
        case SDLK_DOWN: keyState.down = KState::PRESSED; break;
        case SDLK_RSHIFT: keyState.rShift = KState::PRESSED; break;
        case SDLK_RCTRL: keyState.rCtrl = KState::PRESSED; break;
        default: break;
        }
    }
    else if (e.type == SDL_KEYUP) {
        switch (e.key.keysym.sym) {
        case SDLK_LEFT: keyState.left = KState::RELEASED; break;
        case SDLK_RIGHT: keyState.right = KState::RELEASED; break;
        case SDLK_UP: keyState.up = KState::RELEASED; break;
        case SDLK_DOWN: keyState.down = KState::RELEASED; break;
        case SDLK_RSHIFT: keyState.rShift = KState::RELEASED; break;
        case SDLK_RCTRL: keyState.rCtrl = KState::RELEASED; break;
        default: break;
        }
    }
    else {
        assert(false);
    }
}

void InputManager::UpdateMouse()
{
    SDL_PumpEvents();

    int x, y;
    uint32_t buttons = SDL_GetMouseState(&x, &y);

    // Left mouse button.
    if (buttons & SDL_BUTTON_LMASK) {
        keyState.mouseLeft = KState::PRESSED;
    }
    else {
        keyState.mouseLeft = KState::RELEASED;
    }
    // Right mouse button.
    if (buttons & SDL_BUTTON_RMASK) {
        keyState.mouseRight = KState::PRESSED;
    }
    else {
        keyState.mouseRight = KState::RELEASED;
    }
}

glm::vec2 InputManager::CursorPosition() const
{
    SDL_PumpEvents();

    int x, y;
    SDL_GetMouseState(&x, &y);

    float mouseX = x / (float)m_windowPixelSize.x;
    float mouseY = y / (float)m_windowPixelSize.y;

    return { 2*mouseX-1, -(2*mouseY-1) };
}