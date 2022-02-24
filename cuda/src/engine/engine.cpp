#include "engine/engine.h"
#include "third_party/imgui/imgui.h"
#include "third_party/imgui/imgui_impl_sdl.h"
#include "third_party/imgui/imgui_impl_opengl3.h"
#include <stdexcept>


Engine::Engine() 
{
    InitSDL();
    CreateWindow();
    InitImgui();
}

Engine::~Engine() 
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();   
    
    SDL_GL_DeleteContext(m_sdlGlContext);
    SDL_DestroyWindow(m_window);
    SDL_Quit();    
}


void Engine::InitSDL() 
{
    if (SDL_Init(SDL_INIT_VIDEO )) {
        throw std::runtime_error("Failed to initialize SDL : " + std::string(SDL_GetError()));
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);    
}

void Engine::CreateWindow() 
{
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_WindowFlags windowFlags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    m_window = SDL_CreateWindow("Dear ImGui SDL2+OpenGL3 example", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, windowFlags);
    if (!m_window) {
        throw std::runtime_error("Failed to create SDL window");
    }
    m_sdlGlContext = SDL_GL_CreateContext(m_window);
    SDL_GL_MakeCurrent(m_window, m_sdlGlContext);
    SDL_GL_SetSwapInterval(1); // Enable vsync
}

void Engine::InitImgui() 
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    const char* glslVersion = "#version 130";
    ImGui_ImplSDL2_InitForOpenGL(m_window, m_sdlGlContext);
    ImGui_ImplOpenGL3_Init(glslVersion);
}

void Engine::Run() 
{
    bool done = false;
    while (!done)
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                done = true;
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(m_window))
                done = true;
        }

        glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        Render();

        SDL_GL_SwapWindow(m_window);
    } 
}

void Engine::Render() 
{    
    // Draw some geometry
    uint32_t* pixels = new uint32_t[1200 * 900];
    for (size_t x = 0; x < 1200; x++) {
        for (size_t y = 0; y < 900; y++) {
            pixels[y * 1200 + x] = 0xFFFFFFFF;
        }
    }
    glDrawPixels(1200, 900, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)pixels);
    delete[] pixels;

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    // Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    bool show = true;
    ImGui::ShowDemoWindow(&show);

    // Rendering
    ImGui::Render();
    //glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}