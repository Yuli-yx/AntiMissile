cmake_minimum_required(VERSION 3.17)
project(antiMissile)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
find_package(OpenCV 4 REQUIRED)

add_subdirectory(StereoMV)

add_executable(antiMissile main.cpp)

target_link_libraries(antiMissile ${OpenCV_LIBS} StereoMV)