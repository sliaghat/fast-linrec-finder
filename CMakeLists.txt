cmake_minimum_required(VERSION 3.18)
project(HankelRecurrenceSolver LANGUAGES CXX CUDA)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA Standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU Architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Find CUDA Toolkit (Required for cuFFT)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

set(SOURCES
    src/main.cpp
    src/RecurrenceHankelSolver.cu
    third_party/lsqrBase.cpp
)

# Create Executable
add_executable(RecurrenceSolver ${SOURCES})

# Link Libraries
target_link_libraries(RecurrenceSolver PRIVATE 
    CUDA::cudart 
    CUDA::cufft
)

if (MSVC)
    target_compile_options(RecurrenceSolver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler "/W3">)
else()
    target_compile_options(RecurrenceSolver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler "-Wall">)
endif()
