# Requirements
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main Project
project(proj VERSION 1.0)
FILE(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "./src/*.cpp")
add_executable(proj ${CPP_SOURCES})
target_include_directories(proj PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Warnings/Errors : there's no reason to ignore warnings in this project.
target_compile_options(proj PUBLIC -Wall -Wextra)
target_compile_options(proj PUBLIC -Werror -Wno-unused-parameter)

# SDL2
find_package(SDL2 REQUIRED)
target_include_directories(proj PUBLIC ${SDL2_INCLUDE_DIRS})
target_link_libraries(proj ${SDL2_LIBRARIES})

# Vulkan
find_package(Vulkan REQUIRED)
target_include_directories(proj PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(proj ${Vulkan_LIBRARIES})

# VkBootstrap
add_subdirectory(submodules/vk-bootstrap)
target_link_libraries(proj vk-bootstrap::vk-bootstrap)

# GLM
find_package(glm REQUIRED)
target_include_directories(proj PUBLIC ${GLM_INCLUDE_DIRS})
target_link_libraries(proj ${GLM_LIBRARIES})

# pthreads (for std::thread)
target_link_libraries(proj pthread)

## Compile shaders
#set(GLSL_VALIDATOR "glslangValidator")
#set(SHADER_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/shaders")
#
## Shaders we will actually pass to glslangValidator
#set(SHADER_MODULES 
#    "${CMAKE_SOURCE_DIR}/shaders/raycaster/main.comp"
#    "${CMAKE_SOURCE_DIR}/shaders/voxelizer/main.comp"
#    "${CMAKE_SOURCE_DIR}/shaders/renderer/triangle.vert"
#    "${CMAKE_SOURCE_DIR}/shaders/renderer/triangle.frag")
#
## All shader source files
#FILE(GLOB_RECURSE SHADER_SOURCES RELATIVE ${CMAKE_SOURCE_DIR}
#    "shaders/*.comp"
#    "shaders/*.vert"
#    "shaders/*.frag")
#
#foreach(SHADER ${SHADER_MODULES})
#    message("Compiling shader module: ${SHADER}")
#    set(SPIRV "${SHADER}.spv")
#    add_custom_command(
#        OUTPUT ${SPIRV}
#        COMMAND ${GLSL_VALIDATOR} -V -I${SHADER_INCLUDE_DIR} ${SHADER} -o ${SPIRV}
#        DEPENDS ${SHADER_SOURCES}) # we recompile the shader modules everytime ANY shader file changes
#    list(APPEND SPIRV_FILES ${SPIRV})
#endforeach(SHADER)
#
#add_custom_target(
#    shaders 
#    ALL # the shaders will be built by default
#    DEPENDS ${SPIRV_FILES})
