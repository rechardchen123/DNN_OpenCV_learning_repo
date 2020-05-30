cmake_minimum_required(VERSION 3.16)
project(section6)

set(CMAKE_CXX_STANDARD 11)

#add_executable(section6 blemish.cpp)
# add_executable(section6 filter.cpp)
add_executable(section6 chroma_keying.cpp)
set(OpenCV_DIR C:/Users/ucesxc0/Downloads/opencv-4.2.0/minGW-build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(section6 ${OpenCV_LIBS})

