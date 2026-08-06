cmake_minimum_required(VERSION 3.22)
project(SEALTorch VERSION 0.2.0 LANGUAGES CXX CUDA)

include(CTest)
set(FIDESLIB_DIR "" CACHE PATH "Directory containing fideslibConfig.cmake")
find_package(fideslib CONFIG REQUIRED HINTS "${FIDESLIB_DIR}")
find_package(OpenMP REQUIRED)

add_library(sealtorch_core STATIC src/inference.cpp)
target_compile_features(sealtorch_core PUBLIC cxx_std_20)
target_include_directories(sealtorch_core PUBLIC src)
target_link_libraries(sealtorch_core PUBLIC fideslib::fideslib OpenMP::OpenMP_CXX)

add_executable(sealtorch_he src/main.cpp)
target_compile_features(sealtorch_he PRIVATE cxx_std_20)
target_link_libraries(sealtorch_he PRIVATE sealtorch_core)
# The installed FIDESlib/OpenFHE archives were built with GCC 15 while CUDA's
# imported target adds GCC 13's search directory first. Pin the matching C++
# runtime after the static archives so CXXABI_1.3.15 symbols resolve correctly.
if(EXISTS "/usr/lib/gcc/x86_64-linux-gnu/15/libstdc++.so")
    target_link_options(sealtorch_he PRIVATE -Wl,--no-as-needed)
    target_link_libraries(sealtorch_he PRIVATE "/usr/lib/gcc/x86_64-linux-gnu/15/libstdc++.so")
endif()

option(SEALTORCH_WARNINGS "Enable compiler warnings" ON)
if(SEALTORCH_WARNINGS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sealtorch_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(sealtorch_he PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
    add_executable(sealtorch_model_test tests/model_test.cpp)
    target_compile_features(sealtorch_model_test PRIVATE cxx_std_20)
    target_link_libraries(sealtorch_model_test PRIVATE sealtorch_core)
    add_test(NAME sealtorch_model_test COMMAND sealtorch_model_test)
endif()
