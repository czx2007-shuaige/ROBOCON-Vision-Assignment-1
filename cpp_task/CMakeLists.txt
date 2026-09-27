# 最低cmake版本
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(video_process)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 头文件搜索路径（include文件夹 + Eigen）
include_directories(include)
include_directories(/usr/include/eigen3)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 添加可执行程序，把两个cpp源码编译进去
add_executable(video_process src/main.cpp src/transform.cpp)

# 链接OpenCV库
target_link_libraries(video_process ${OpenCV_LIBS})
