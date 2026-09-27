cmake_minimum_required(VERSION 3.26)
project(Fluid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The Vulkan SDK installer places SDL3's CMake package and GLM headers here.
if(DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK_ROOT "$ENV{VULKAN_SDK}")
elseif(DEFINED Vulkan_INCLUDE_DIRS)
    get_filename_component(VULKAN_SDK_ROOT "${Vulkan_INCLUDE_DIRS}" DIRECTORY)
else()
    message(FATAL_ERROR "VULKAN_SDK is not set. Install the Vulkan SDK or set VULKAN_SDK to its installation directory.")
endif()

find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED HINTS "${VULKAN_SDK_ROOT}/cmake")

find_package(CUDAToolkit REQUIRED)

add_executable(Fluid
    src/main.cpp
    src/App.cpp
    src/vulkan/VulkanContext.cpp
    src/vulkan/Renderer.cpp
    src/cuda/CudaContext.cpp
    src/cuda/Simulation.cu
    src/interop/SharedBuffer.cpp
    src/interop/InteropSemaphore.cpp
)
target_compile_definitions(Fluid PRIVATE VK_USE_PLATFORM_WIN32_KHR NOMINMAX)
target_include_directories(Fluid PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(Fluid PRIVATE Vulkan::Vulkan SDL3::SDL3 CUDA::cudart_static)

set_target_properties(Fluid PROPERTIES
    CUDA_ARCHITECTURES 86
)

if(MSVC)
    target_compile_options(Fluid PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
endif()
