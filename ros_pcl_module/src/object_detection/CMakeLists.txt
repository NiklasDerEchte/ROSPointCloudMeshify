cmake_minimum_required(VERSION 3.8)
project(object_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(PCL REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Build object detection
add_executable(object_detection_node src/object_detection_node.cpp)
ament_target_dependencies(object_detection_node rclcpp sensor_msgs pcl_conversions pcl_ros)
target_link_libraries(object_detection_node
  ${PCL_LIBRARIES}
)
install(TARGETS object_detection_node
  DESTINATION lib/${PROJECT_NAME})

# Build visualizer
add_executable(object_detection_visualizer_node src/object_detection_visualizer.cpp)
ament_target_dependencies(object_detection_visualizer_node rclcpp sensor_msgs pcl_conversions pcl_ros visualization_msgs)
target_link_libraries(object_detection_visualizer_node
  ${PCL_LIBRARIES}
)
install(TARGETS object_detection_visualizer_node
  DESTINATION lib/${PROJECT_NAME})

# Build geometry visualizer
add_executable(object_detection_geometryc_visualizer_node src/object_detection_geometryc_visualizer.cpp)
ament_target_dependencies(object_detection_geometryc_visualizer_node rclcpp sensor_msgs pcl_conversions pcl_ros visualization_msgs)
target_link_libraries(object_detection_geometryc_visualizer_node
  ${PCL_LIBRARIES}
)
install(TARGETS object_detection_geometryc_visualizer_node
  DESTINATION lib/${PROJECT_NAME})

# Build mesh creator
add_executable(object_detection_mesh_creator_node src/object_detection_mesh_creator.cpp)
ament_target_dependencies(object_detection_mesh_creator_node rclcpp sensor_msgs pcl_conversions pcl_ros visualization_msgs)
target_link_libraries(object_detection_mesh_creator_node
  ${PCL_LIBRARIES}
)
install(TARGETS object_detection_mesh_creator_node
  DESTINATION lib/${PROJECT_NAME})

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
