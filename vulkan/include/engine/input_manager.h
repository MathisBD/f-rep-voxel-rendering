#pragma once
#include <SDL2/SDL.h>
#include <glm/glm.hpp>



class InputManager
{
public:
    enum class KState
    {
        PRESSED,
        RELEASED
    };

    struct {
        KState left = KState::RELEASED;
        KState right = KState::RELEASED;
        KState up = KState::RELEASED;
        KState down = KState::RELEASED;
        KState rShift = KState::RELEASED;
        KState rCtrl = KState::RELEASED;
        KState mouseLeft = KState::RELEASED;
        KState mouseRight = KState::RELEASED;
    } keyState;

    void Init(const glm::uvec2& windowPixelSize);
    void UpdateKey(const SDL_Event& e);
    void UpdateMouse();

    // Axis orientation :
    // x : left to right.
    // y : bottom to top.
    // The center of the screen is at (0, 0).
    // The borders of the screen are at -1 and +1.
    glm::vec2 CursorPosition() const;
private:
    glm::uvec2 m_windowPixelSize;
};