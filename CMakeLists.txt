# Requirements
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main Project
project(proj VERSION 1.0)
FILE(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "./src/*.cpp")
add_executable(proj ${CPP_SOURCES})
target_include_directories(proj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Warnings/Errors : there's no reason to ignore warnings in this project.
target_compile_options(proj PRIVATE -Wall -Wextra)
target_compile_options(proj PRIVATE -Werror)

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
