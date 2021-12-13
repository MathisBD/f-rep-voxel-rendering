#include "engine/vk_engine.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>


void VkEngine::Init() 
{
    SDL_Init(SDL_INIT_VIDEO);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);
	m_window = SDL_CreateWindow(
		"Vulkan Engine",            //window title
		SDL_WINDOWPOS_UNDEFINED,    // window position x (don't care)
		SDL_WINDOWPOS_UNDEFINED,    // window position y (don't care)
		m_windowExtent.width,       // window width in pixels
		m_windowExtent.height,      // window height in pixels
		window_flags 
	);

    m_isInitialized = true;    
}

void VkEngine::Run() 
{
    SDL_Event e;
    bool quit = false;

    while (!quit) {
        // poll SDL events
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }
        // main engine logic is here
        Draw();
    }
}

void VkEngine::Draw() 
{
    
}

void VkEngine::Cleanup() 
{
    if (m_isInitialized) {
        SDL_DestroyWindow(m_window);
    }    
}