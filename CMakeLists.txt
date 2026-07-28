cmake_minimum_required(VERSION 3.22)

project(SEALTorch VERSION 0.1.0 LANGUAGES CXX)

# Build each provider as an isolated target.  The CPU worker uses Microsoft
# SEAL directly; the CUDA worker uses FIDESlib directly.  Keeping the two
# implementations separate is intentional: neither provider is disguised as
# the other provider's API.
option(SEALTORCH_BUILD_SEAL_BACKEND "Build the Microsoft SEAL CPU worker" ON)
option(SEALTORCH_BUILD_FIDESLIB_BACKEND "Build the FIDESlib CUDA worker" ON)
set(SEAL_DIR "$ENV{HOME}/.local/lib/cmake/SEAL-4.4" CACHE PATH
    "Directory containing SEALConfig.cmake")
set(FIDESLIB_DIR "" CACHE PATH "Directory containing fideslibConfig.cmake")

if(SEALTORCH_BUILD_SEAL_BACKEND)
    # The SEAL CMake package treats minor versions as incompatible. The
    # evaluator uses the stable public API, so accept the installed SEAL
    # release instead of pinning an obsolete 4.3 package directory.
    find_package(SEAL CONFIG REQUIRED HINTS "${SEAL_DIR}")
endif()
if(SEALTORCH_BUILD_FIDESLIB_BACKEND)
    find_package(fideslib CONFIG REQUIRED HINTS "${FIDESLIB_DIR}")
endif()
if(NOT SEALTORCH_BUILD_SEAL_BACKEND AND NOT SEALTORCH_BUILD_FIDESLIB_BACKEND)
    message(FATAL_ERROR "Enable at least one HE backend")
endif()
find_package(OpenMP REQUIRED)

set(SEALTORCH_SOURCES
    src/SEALTorch/math.cpp
    src/SEALTorch/evaluator.cpp
    src/SEALTorch/metrics.cpp
    src/SEALTorch/inference.cpp)

function(add_seal_worker)
    add_library(SEALTorch_cpu STATIC ${SEALTORCH_SOURCES})
    target_compile_features(SEALTorch_cpu PUBLIC cxx_std_20)
    target_include_directories(SEALTorch_cpu
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>)
    target_link_libraries(SEALTorch_cpu PUBLIC SEAL::seal OpenMP::OpenMP_CXX)

    add_executable(sealtorch_gui_cpu src/main.cpp)
    target_compile_features(sealtorch_gui_cpu PRIVATE cxx_std_20)
    target_link_libraries(sealtorch_gui_cpu PRIVATE SEALTorch_cpu)
endfunction()

if(SEALTORCH_BUILD_SEAL_BACKEND)
    add_seal_worker()
endif()
if(SEALTORCH_BUILD_FIDESLIB_BACKEND)
    # This executable deliberately does not link SEALTorch_cpu.  Its source
    # names and uses FIDESlib types and operations directly.
    add_executable(sealtorch_gui_cuda src/fides_main.cpp)
    target_compile_features(sealtorch_gui_cuda PRIVATE cxx_std_20)
    target_include_directories(sealtorch_gui_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(sealtorch_gui_cuda PRIVATE fideslib::fideslib OpenMP::OpenMP_CXX)
endif()

# Keep a conventional library alias for the native SEAL library.
if(SEALTORCH_BUILD_SEAL_BACKEND AND NOT SEALTORCH_BUILD_FIDESLIB_BACKEND)
    add_library(SEALTorch::SEALTorch ALIAS SEALTorch_cpu)
endif()
