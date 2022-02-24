#pragma once
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>


class Engine
{
public:
    Engine();
    ~Engine();
    void Run();
private:
    SDL_Window* m_window = nullptr;
    SDL_GLContext m_sdlGlContext;
   
    void InitSDL();
    void InitImgui();
    void CreateWindow();

    void Render();
};