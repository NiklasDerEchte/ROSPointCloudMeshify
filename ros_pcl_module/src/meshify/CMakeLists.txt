cmake_minimum_required(VERSION 3.8)
project(meshify)

# An empty build type compiles the numeric kernels without optimization and
# makes the stage timings incomparable. Debug remains available when requested.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "meshify: CMAKE_BUILD_TYPE was empty; defaulting to Release. Pass -DCMAKE_BUILD_TYPE=Debug to override.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(WARNING "meshify: Debug build. Stage timings are not comparable to Release.")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
  )
endif()

# find dependencies
find_package(PCL REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

set(NODES
  marker_array_node
  marker_array_node_2
  float_array_node
  float_array_unity_plane_node
)

foreach(node ${NODES})
  add_executable(${node} src/${node}.cpp)
  ament_target_dependencies(${node} rclcpp sensor_msgs std_msgs geometry_msgs pcl_conversions pcl_ros visualization_msgs)
  target_link_libraries(${node} ${PCL_LIBRARIES})
  install(TARGETS ${node} DESTINATION lib/${PROJECT_NAME})
endforeach()

# --- fast grid mesh (cpu / openmp / optional cuda) --------------------------
option(MESHIFY_USE_CUDA "Build the CUDA backend when a CUDA toolkit is available" ON)
set(MESHIFY_HAS_CUDA 0)
if(MESHIFY_USE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    # nvcc from CUDA 12.0–12.3 cannot parse GCC 13 headers. Prefer g++-12
    # when it is installed; newer toolkits keep the default host compiler.
    if(NOT CMAKE_CUDA_HOST_COMPILER
       AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
       AND CUDAToolkit_VERSION VERSION_LESS 12.4)
      find_program(_meshify_gcc12 NAMES g++-12)
      if(_meshify_gcc12)
        set(CMAKE_CUDA_HOST_COMPILER "${_meshify_gcc12}" CACHE FILEPATH "Host compiler for nvcc")
        message(STATUS "meshify: using ${CMAKE_CUDA_HOST_COMPILER} as the CUDA host compiler")
      endif()
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
      set(MESHIFY_HAS_CUDA 1)
    else()
      message(STATUS "meshify: CUDA toolkit headers were found but the CUDA compiler is not usable. cpu/openmp only.")
    endif()
  else()
    message(STATUS "meshify: CUDA toolkit not found. cpu/openmp only.")
  endif()
else()
  message(STATUS "meshify: MESHIFY_USE_CUDA=OFF. cpu/openmp only.")
endif()

add_library(fast_grid_mesh_core STATIC src/fast_grid_mesh.cpp)
target_include_directories(fast_grid_mesh_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(fast_grid_mesh_core PUBLIC cxx_std_17)
target_compile_options(fast_grid_mesh_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
# OpenMP is linked plainly (ament_target_dependencies uses the plain signature).
# -fopenmp still comes from the OpenMP::OpenMP_CXX interface.
ament_target_dependencies(fast_grid_mesh_core sensor_msgs)
target_link_libraries(fast_grid_mesh_core Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_definitions(fast_grid_mesh_core PUBLIC
  MESHIFY_HAS_OPENMP=1
  MESHIFY_HAS_CUDA=${MESHIFY_HAS_CUDA})

if(MESHIFY_HAS_CUDA)
  if(NOT MESHIFY_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
      set(_meshify_cuda_archs 75;80;86;89;90;100)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
      set(_meshify_cuda_archs 75;80;86;89;90)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
      set(_meshify_cuda_archs 75;80;86;89)
    else()
      set(_meshify_cuda_archs 75;80;86)
    endif()
    set(MESHIFY_CUDA_ARCHITECTURES "${_meshify_cuda_archs}" CACHE STRING
      "CUDA architectures for fast_grid_mesh, e.g. 86 or 75;86;89")
  endif()
  message(STATUS "meshify: CUDA backend enabled (toolkit ${CUDAToolkit_VERSION}, archs ${MESHIFY_CUDA_ARCHITECTURES})")

  add_library(fast_grid_mesh_cuda STATIC src/fast_grid_mesh_cuda.cu)
  target_include_directories(fast_grid_mesh_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  set_target_properties(fast_grid_mesh_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${MESHIFY_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_options(fast_grid_mesh_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  target_link_libraries(fast_grid_mesh_cuda PUBLIC CUDA::cudart fast_grid_mesh_core)
  target_compile_definitions(fast_grid_mesh_cuda PRIVATE MESHIFY_HAS_CUDA=1 MESHIFY_HAS_OPENMP=1)
endif()

add_executable(fast_grid_mesh_node src/fast_grid_mesh_node.cpp)
target_compile_features(fast_grid_mesh_node PUBLIC cxx_std_17)
target_link_libraries(fast_grid_mesh_node fast_grid_mesh_core OpenMP::OpenMP_CXX)
ament_target_dependencies(fast_grid_mesh_node rclcpp sensor_msgs std_msgs geometry_msgs)
if(MESHIFY_HAS_CUDA)
  target_link_libraries(fast_grid_mesh_node fast_grid_mesh_cuda)
endif()
install(TARGETS fast_grid_mesh_node DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

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

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(fast_grid_mesh_test test/test_fast_grid_mesh.cpp)
  if(TARGET fast_grid_mesh_test)
    target_include_directories(fast_grid_mesh_test PRIVATE ${PCL_INCLUDE_DIRS})
    target_compile_options(fast_grid_mesh_test PRIVATE -ffp-contract=off)
    target_link_libraries(fast_grid_mesh_test fast_grid_mesh_core OpenMP::OpenMP_CXX ${PCL_LIBRARIES})
    ament_target_dependencies(fast_grid_mesh_test sensor_msgs pcl_conversions)
    if(MESHIFY_HAS_CUDA)
      target_link_libraries(fast_grid_mesh_test fast_grid_mesh_cuda)
    endif()
  endif()
endif()

ament_package()
