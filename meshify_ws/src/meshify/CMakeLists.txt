cmake_minimum_required(VERSION 3.8)
project(meshify)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # CXX only. nvcc rejects some of these host flags.
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
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

set(NODES
  float_array_node
  float_array_pcl_node
)

foreach(node ${NODES})
  add_executable(${node} src/${node}.cpp)
  ament_target_dependencies(${node} rclcpp sensor_msgs std_msgs pcl_conversions pcl_ros)
  target_link_libraries(${node} ${PCL_LIBRARIES})
  install(TARGETS ${node} DESTINATION lib/${PROJECT_NAME})
endforeach()

# CUDA is required for this node only. The rest of the package still builds.
find_package(CUDAToolkit QUIET)
set(MESHIFY_CUDA_NODE OFF)
if(CUDAToolkit_FOUND)
  # Pick an older GCC as nvcc host compiler when the default one is too new:
  #   CUDA < 11.6: <functional> of GCC 11+ fails ("parameter packs not expanded"), use g++-10
  #   CUDA 12.0-12.3: cannot parse GCC 13 headers, use g++-12
  # Set CMAKE_CUDA_HOST_COMPILER yourself to override.
  set(_meshify_host_gcc "")
  if(NOT CMAKE_CUDA_HOST_COMPILER AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CUDAToolkit_VERSION VERSION_LESS 11.6 AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
      set(_meshify_host_gcc g++-10)
    elseif(CUDAToolkit_VERSION VERSION_LESS 12.4 AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
      set(_meshify_host_gcc g++-12)
    endif()
  endif()
  if(_meshify_host_gcc)
    find_program(_meshify_host_gcc_path NAMES ${_meshify_host_gcc})
    if(_meshify_host_gcc_path)
      # CMake 3.22 reads the host compiler from CUDAHOSTCXX when it detects the CUDA compiler.
      set(ENV{CUDAHOSTCXX} "${_meshify_host_gcc_path}")
      set(CMAKE_CUDA_HOST_COMPILER "${_meshify_host_gcc_path}" CACHE FILEPATH "Host compiler for nvcc")
      message(STATUS "meshify: using ${CMAKE_CUDA_HOST_COMPILER} as the CUDA host compiler")
    else()
      message(WARNING "meshify: CUDA ${CUDAToolkit_VERSION} needs ${_meshify_host_gcc} as host compiler, "
                      "install it or set CMAKE_CUDA_HOST_COMPILER")
    endif()
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(MESHIFY_CUDA_NODE ON)
  endif()
endif()

if(MESHIFY_CUDA_NODE)
  find_package(OpenMP REQUIRED)

  if(NOT MESHIFY_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
      set(_meshify_cuda_archs 75;80;86;89;90)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
      set(_meshify_cuda_archs 75;80;86;89)
    else()
      set(_meshify_cuda_archs 75;80;86)
    endif()
    set(MESHIFY_CUDA_ARCHITECTURES "${_meshify_cuda_archs}" CACHE STRING
      "CUDA architectures for float_array_cuda_node, e.g. 86")
  endif()
  message(STATUS "meshify: building float_array_cuda_node (toolkit ${CUDAToolkit_VERSION}, archs ${MESHIFY_CUDA_ARCHITECTURES})")

  add_library(float_array_cuda STATIC src/float_array_cuda.cu)
  set_target_properties(float_array_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${MESHIFY_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(float_array_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  target_link_libraries(float_array_cuda PUBLIC CUDA::cudart)

  add_executable(float_array_cuda_node src/float_array_cuda_node.cpp)
  ament_target_dependencies(float_array_cuda_node rclcpp sensor_msgs std_msgs)
  target_link_libraries(float_array_cuda_node float_array_cuda OpenMP::OpenMP_CXX)
  install(TARGETS float_array_cuda_node DESTINATION lib/${PROJECT_NAME})
else()
  message(STATUS "meshify: CUDA toolkit not found, skipping float_array_cuda_node")
endif()

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
