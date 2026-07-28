cmake_minimum_required(VERSION 3.22)

project(SEALTorch VERSION 0.1.0 LANGUAGES CXX)

# Build each provider as an isolated target. Both binaries can coexist, which
# lets the web worker choose CPU SEAL or CUDA FIDESlib for every experiment.
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

function(add_he_worker suffix uses_fideslib)
    add_library(SEALTorch_${suffix} STATIC ${SEALTORCH_SOURCES})
    target_compile_features(SEALTorch_${suffix} PUBLIC cxx_std_20)
    target_include_directories(SEALTorch_${suffix}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
            $<INSTALL_INTERFACE:include>)
    target_compile_definitions(SEALTorch_${suffix} PUBLIC SEALTORCH_USE_FIDESLIB=${uses_fideslib})
    if(uses_fideslib)
        target_link_libraries(SEALTorch_${suffix} PUBLIC fideslib::fideslib)
    else()
        target_link_libraries(SEALTorch_${suffix} PUBLIC SEAL::seal)
    endif()
    # Static FIDESlib uses OpenMP, so keep the runtime after its archive.
    target_link_libraries(SEALTorch_${suffix} PUBLIC OpenMP::OpenMP_CXX)

    add_executable(sealtorch_gui_${suffix} src/main.cpp)
    target_compile_features(sealtorch_gui_${suffix} PRIVATE cxx_std_20)
    target_link_libraries(sealtorch_gui_${suffix} PRIVATE SEALTorch_${suffix})
endfunction()

if(SEALTORCH_BUILD_SEAL_BACKEND)
    add_he_worker(cpu 0)
endif()
if(SEALTORCH_BUILD_FIDESLIB_BACKEND)
    add_he_worker(cuda 1)
endif()

# Keep a conventional library alias for consumers that build exactly one
# provider. A dual-provider build intentionally uses the suffixed targets.
if(SEALTORCH_BUILD_SEAL_BACKEND AND NOT SEALTORCH_BUILD_FIDESLIB_BACKEND)
    add_library(SEALTorch::SEALTorch ALIAS SEALTorch_cpu)
elseif(SEALTORCH_BUILD_FIDESLIB_BACKEND AND NOT SEALTORCH_BUILD_SEAL_BACKEND)
    add_library(SEALTorch::SEALTorch ALIAS SEALTorch_cuda)
endif()
