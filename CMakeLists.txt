cmake_minimum_required(VERSION 3.10)
project(kf_gins)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(kf_gins_node
    src/kf_gins_node.cpp
    src/kf-gins/gi_engine.cpp
    src/kf-gins/insmech.cpp
)

add_executable(kf_gins_csv_replay
    src/kf_gins_csv_replay.cpp
    src/kf-gins/gi_engine.cpp
    src/kf-gins/insmech.cpp
)

target_include_directories(kf_gins_node PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

target_include_directories(kf_gins_csv_replay PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(kf_gins_node
    Eigen3::Eigen
)

target_link_libraries(kf_gins_csv_replay
    Eigen3::Eigen
)

ament_target_dependencies(kf_gins_node
    rclcpp
    sensor_msgs
    nav_msgs
    geometry_msgs
    tf2
    tf2_ros
)

ament_target_dependencies(kf_gins_csv_replay
    rclcpp
    sensor_msgs
    nav_msgs
    geometry_msgs
    tf2
    tf2_ros
)

install(TARGETS kf_gins_node
    DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS kf_gins_csv_replay
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
)

# install(DIRECTORY dataset
#     DESTINATION share/${PROJECT_NAME}
# )

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
    scripts/tune_csv_replay.py
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
