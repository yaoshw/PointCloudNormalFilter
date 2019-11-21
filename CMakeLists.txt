cmake_minimum_required(VERSION 3.14)
project(PointCloudNormalFilter)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(PointCloudNormalFilter main.cpp)

target_link_libraries (
        PointCloudNormalFilter
        ${PCL_LIBRARIES}
)