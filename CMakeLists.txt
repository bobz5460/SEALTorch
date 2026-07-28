cmake_minimum_required(VERSION 3.22)

project(SEALTorch VERSION 0.1.0 LANGUAGES CXX)

# SEALTorch is one research library with two native providers.  Every normal
# build contains both: Microsoft SEAL for the CPU baseline and FIDESlib for
# CUDA.  The application chooses the provider at runtime.
set(SEAL_DIR "$ENV{HOME}/.local/lib/cmake/SEAL-4.4" CACHE PATH
    "Directory containing SEALConfig.cmake")
set(FIDESLIB_DIR "" CACHE PATH "Directory containing fideslibConfig.cmake")
find_package(SEAL CONFIG REQUIRED HINTS "${SEAL_DIR}")
find_package(fideslib CONFIG REQUIRED HINTS "${FIDESLIB_DIR}")
find_package(OpenMP REQUIRED)

add_library(SEALTorch STATIC
    src/SEALTorch/math.cpp
    src/SEALTorch/evaluator.cpp
    src/SEALTorch/metrics.cpp
    src/SEALTorch/inference.cpp
    src/SEALTorch/fides_inference.cpp)
add_library(SEALTorch::SEALTorch ALIAS SEALTorch)

target_compile_features(SEALTorch PUBLIC cxx_std_20)
target_include_directories(SEALTorch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(SEALTorch PUBLIC
    SEAL::seal
    fideslib::fideslib
    OpenMP::OpenMP_CXX)

add_executable(sealtorch_gui src/main.cpp)
target_compile_features(sealtorch_gui PRIVATE cxx_std_20)
target_link_libraries(sealtorch_gui PRIVATE SEALTorch)
