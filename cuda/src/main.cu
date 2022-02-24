#include <SDL2/SDL.h>


int main()
{
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("SDL Test", 100, 100, 1200, 900, SDL_WINDOW_SHOWN);
    SDL_Surface* surface = SDL_GetWindowSurface(window);

    SDL_FillRect(surface, NULL, SDL_MapRGB(surface->format, 0xFF, 0xFF, 0xFF));
    
    for (uint32_t x = 0; x < surface->w; x++) {
        for (uint32_t y = 0; y < surface->h; y++) {
            if (x*x + y*y < 500*500) {
                size_t idx = y * surface->pitch + x * surface->format->BytesPerPixel;
                uint32_t color = SDL_MapRGB(surface->format, 0xFF, 0xFF, 0x00);
                memcpy((void*)((size_t)surface->pixels + idx), &color, surface->format->BytesPerPixel);
            }
        }
    }

    SDL_UpdateWindowSurface(window);

    SDL_Event event;
    bool quit = false;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            }
            else if (event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
                // handle key input
            }
        }
        // update mouse position


    }

    SDL_DestroyWindow(window);

    SDL_Quit();
    return 0;
}