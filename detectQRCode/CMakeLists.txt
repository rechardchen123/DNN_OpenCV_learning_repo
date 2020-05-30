cmake_minimum_required(VERSION 3.16)
project(detectQRCode)

set(CMAKE_CXX_STANDARD 11)

add_executable(detectQRcode main.cpp)
FIND_PACKAGE(OpenCV REQUIRED)
IF (OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(detectQRcode ${OpenCV_LIBS})
ELSE (OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV library not found")
ENDIF (OpenCV_FOUND)