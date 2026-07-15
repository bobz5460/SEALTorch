cmake_minimum_required(VERSION 3.22)

project(SEALTorch VERSION 0.1.0 LANGUAGES CXX)

# SEAL is installed separately and is only a dependency of this project.
set(SEAL_DIR "$ENV{HOME}/.local/seal/lib/cmake/SEAL-4.3" CACHE PATH
    "Directory containing SEALConfig.cmake")

find_package(SEAL 4.3.3 EXACT CONFIG REQUIRED
    HINTS "${SEAL_DIR}")

add_library(SEALTorch STATIC
    src/SEALTorch/math.cpp
    src/SEALTorch/evaluator.cpp)
add_library(SEALTorch::SEALTorch ALIAS SEALTorch)

target_compile_features(SEALTorch PUBLIC cxx_std_17)
target_include_directories(SEALTorch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(SEALTorch PUBLIC SEAL::seal)

find_package(X11 REQUIRED)

add_executable(sealtorch_gui src/main.cpp)
target_compile_features(sealtorch_gui PRIVATE cxx_std_17)
target_link_libraries(sealtorch_gui PRIVATE SEALTorch::SEALTorch X11::X11)

add_executable(sealtorch_tiny_test src/tiny_inference_test.cpp)
target_compile_features(sealtorch_tiny_test PRIVATE cxx_std_17)
target_link_libraries(sealtorch_tiny_test PRIVATE SEALTorch::SEALTorch)
