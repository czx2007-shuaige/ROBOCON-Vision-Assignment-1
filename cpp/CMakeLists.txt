cmake_minimum_required(VERSION 3.16)
project(cpp_video_process)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(video_app src/main.cpp src/transform.cpp)
target_link_libraries(video_app ${OpenCV_LIBS})
