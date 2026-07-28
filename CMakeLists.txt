cmake_minimum_required(VERSION 3.22)

project(SEALTorch VERSION 0.1.0 LANGUAGES CXX)

# FIDESlib is installed separately (normally from ../FIDESlib).
set(FIDESLIB_DIR "" CACHE PATH "Directory containing fideslibConfig.cmake")
find_package(fideslib CONFIG REQUIRED HINTS "${FIDESLIB_DIR}")
find_package(OpenMP REQUIRED)

add_library(SEALTorch STATIC
    src/SEALTorch/math.cpp
    src/SEALTorch/evaluator.cpp
    src/SEALTorch/metrics.cpp
    src/SEALTorch/inference.cpp)
add_library(SEALTorch::SEALTorch ALIAS SEALTorch)

target_compile_features(SEALTorch PUBLIC cxx_std_20)
target_include_directories(SEALTorch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(SEALTorch PUBLIC fideslib::fideslib OpenMP::OpenMP_CXX)

find_package(X11 REQUIRED)

add_executable(sealtorch_gui src/main.cpp)
target_compile_features(sealtorch_gui PRIVATE cxx_std_20)
target_link_libraries(sealtorch_gui PRIVATE SEALTorch::SEALTorch X11::X11)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/trace_main.cpp")
    add_executable(sealtorch_trace src/trace_main.cpp)
    target_compile_features(sealtorch_trace PRIVATE cxx_std_17)
    target_link_libraries(sealtorch_trace PRIVATE SEALTorch::SEALTorch)
endif()

# Add your application or tests here as they are created. Keeping this file
# small means the full FIDESlib source tree is never configured as part of your
# project.
