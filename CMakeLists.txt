cmake_minimum_required(VERSION 3.22)

project(SEALTorch VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(SEAL_DIR "$ENV{HOME}/.local/lib/cmake/SEAL-4.4" CACHE PATH
    "Directory containing SEALConfig.cmake")
set(FIDESLIB_DIR "" CACHE PATH "Directory containing fideslibConfig.cmake")
find_package(SEAL CONFIG REQUIRED HINTS "${SEAL_DIR}")

option(SEALTORCH_ENABLE_CUDA "Build the optional FIDESlib CUDA provider" ON)
if(SEALTORCH_ENABLE_CUDA)
    find_package(fideslib CONFIG QUIET HINTS "${FIDESLIB_DIR}")
endif()

if(fideslib_FOUND)
    find_package(OpenMP REQUIRED)
    set(SEALTORCH_CUDA_SOURCE src/SEALTorch/cuda_ciphertext_inference.cpp)
    message(STATUS "SEALTorch CUDA provider enabled")
else()
    set(SEALTORCH_CUDA_SOURCE src/SEALTorch/cuda_ciphertext_inference_stub.cpp)
    message(STATUS "SEALTorch CUDA provider disabled (FIDESlib not found)")
endif()

add_library(SEALTorch STATIC
    src/SEALTorch/math.cpp
    src/SEALTorch/evaluator.cpp
    src/SEALTorch/metrics.cpp
    src/SEALTorch/seal_ciphertext_backend.cpp
    src/SEALTorch/ciphertext_inference.cpp
    ${SEALTORCH_CUDA_SOURCE})
add_library(SEALTorch::SEALTorch ALIAS SEALTorch)

target_compile_features(SEALTorch PUBLIC cxx_std_20)
target_include_directories(SEALTorch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(SEALTorch PUBLIC SEAL::seal)
if(fideslib_FOUND)
    target_link_libraries(SEALTorch PRIVATE
        fideslib::fideslib
        OpenMP::OpenMP_CXX)
endif()

option(SEALTORCH_WARNINGS "Enable helpful compiler warnings" ON)
if(SEALTORCH_WARNINGS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(SEALTorch PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(sealtorch_gui src/main.cpp)
target_compile_features(sealtorch_gui PRIVATE cxx_std_20)
target_link_libraries(sealtorch_gui PRIVATE SEALTorch)
if(SEALTORCH_WARNINGS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sealtorch_gui PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
    add_executable(sealtorch_model_test tests/model_test.cpp)
    target_compile_features(sealtorch_model_test PRIVATE cxx_std_20)
    target_link_libraries(sealtorch_model_test PRIVATE SEALTorch)
    if(SEALTORCH_WARNINGS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(sealtorch_model_test PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME sealtorch_model_test COMMAND sealtorch_model_test)

    add_executable(sealtorch_inference_test tests/inference_test.cpp)
    target_compile_features(sealtorch_inference_test PRIVATE cxx_std_20)
    target_link_libraries(sealtorch_inference_test PRIVATE SEALTorch)
    if(SEALTORCH_WARNINGS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(sealtorch_inference_test PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME sealtorch_inference_test COMMAND sealtorch_inference_test)
endif()
