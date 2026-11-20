cmake_minimum_required(VERSION 3.22)
project(meshify)
enable_language(CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(ament_cmake_DIR "/opt/ros/humble/share/ament_cmake/cmake")
set(ament_cmake_libraries_DIR "/opt/ros/humble/share/ament_cmake_libraries/cmake")
set(ament_cmake_core_DIR "/opt/ros/humble/share/ament_cmake_core/cmake")
#set(ament_cmake_export_dependencies_DIR "/opt/ros/humble/share/ament_cmake_export_dependencies/cmake")
#set(ament_cmake_export_include_directories_DIR "/opt/ros/humble/share/ament_cmake_export_include_directories/cmake")
#set(ament_cmake_export_libraries_DIR "/opt/ros/humble/share/ament_cmake_export_libraries/cmake")

set(BOOST_ROOT /usr/lib/x86_64-linux-gnu/cmake/Boost-1.74.0)
set(CMAKE_PREFIX_PATH "/usr/local")
set(PCL_DIR "/usr/local/share/pcl-1.14")
set(Boost_NO_BOOST_CMAKE ON)
set(PCL_PKG_CONFIG_PATH "/usr/local/lib/pkgconfig")
set(Boost_USE_STATIC_LIBS ON)

include_directories(/usr/local/share/pcl-1.14)
link_directories(/usr/local/lib)

message(STATUS "PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL_LIBRARY_DIRS: ${PCL_LIBRARY_DIRS}")
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")

find_package(PkgConfig)

include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

# find dependencies
find_package(PCL REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_core REQUIRED)
find_package(ament_cmake_libraries REQUIRED)
#find_package(ament_cmake_export_dependencies REQUIRED)
#find_package(ament_cmake_export_include_directories REQUIRED)
#find_package(ament_cmake_export_libraries REQUIRED)


find_package(Boost REQUIRED COMPONENTS date_time system filesystem)

set(NODES 
  mesh_node
  marker_array_node
)

foreach(node ${NODES})
  add_executable(${node} src/${node}.cpp)
  ament_target_dependencies(${node} rclcpp sensor_msgs pcl_conversions pcl_ros visualization_msgs geometry_msgs shape_msgs)
  target_link_libraries(${node} 
    ${PCL_LIBRARIES} 
  )
  install(TARGETS ${node} DESTINATION lib/${PROJECT_NAME})
endforeach()

add_executable(gpu_marker_array_node
    src/gpu_marker_array_node.cpp
    src/plane_segmentation.cu
)
ament_target_dependencies(gpu_marker_array_node rclcpp sensor_msgs pcl_conversions pcl_ros visualization_msgs geometry_msgs shape_msgs)
set_target_properties(gpu_marker_array_node PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_marker_array_node
    ${PCL_LIBRARIES}
    ${CUDA_LIBRARIES}
)
install(TARGETS gpu_marker_array_node
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
