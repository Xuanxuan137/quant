cmake_minimum_required(VERSION 3.10)
project(quant)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3 -W -Wall -fno-elide-constructors -lpthread -fopenmp")
# -fno-elide-constructors的作用：避免gcc对函数返回对象进行优化。
# gcc优化后会导致使用函数返回对象创建新对象时，不会调用拷贝构造函数

# Add the source in project root directory
aux_source_directory(. DIR_SRCS)

# Add header file include directories
include_directories(
        /usr/include/c++/8
        ${PROJECT_SOURCE_DIR}/nn
        ${PROJECT_SOURCE_DIR}/tensor
        ${PROJECT_SOURCE_DIR}/util
        /usr/local/include/opencv4
        /opt/OpenBLAS/include
)

link_directories(
        /usr/local/lib
        /opt/OpenBLAS/lib
)

# Add block directories
add_subdirectory(nn)
add_subdirectory(tensor)
add_subdirectory(util)

# Target
add_executable(
        quant quant.cpp
)
target_link_libraries(
        quant tensor nn util opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs openblas m pthread
)
