# requirements
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Project
project(proj LANGUAGES CUDA CXX)
FILE(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "./src/*.cpp" "./src/*.cu")
add_executable(proj ${CPP_SOURCES})
target_include_directories(proj PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Warnings/Errors
#target_compile_options(proj PUBLIC --Wreorder)
#target_compile_options(proj PUBLIC --Werror cross-execution-space-call)

# Enable CUDA profiling data
target_compile_options(proj PUBLIC -g -lineinfo)

# SDL2
find_package(SDL2 REQUIRED)
target_include_directories(proj PUBLIC ${SDL2_INCLUDE_DIRS})
target_link_libraries(proj ${SDL2_LIBRARIES})

