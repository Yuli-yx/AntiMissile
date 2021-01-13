cmake_minimum_required(VERSION 3.17)
project(antiMissile)

add_compile_definitions(EXAMPLE_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 14)

include_directories(StereoMV)
add_subdirectory(StereoMV)
set(INC_DIR "/home/willzhuyx/antiMissile/StereoMV/include")
set(LINK_DIR "/home/willzhuyx/antiMissile/StereoMV")

include_directories(${INC_DIR})
link_directories(${LINK_DIR})

find_package(OpenCV 4 REQUIRED)

add_executable(antiMissile main.cpp)

target_link_libraries(antiMissile ${OpenCV_LIBS} StereoMV)

