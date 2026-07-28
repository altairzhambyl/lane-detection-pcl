cmake_minimum_required(VERSION 3.8)
project(ransac_seg_cpp)

# The Python package is the reference implementation; this one exists to carry
# the per-sweep hot path, so it is built for speed and warned at hard.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic -O2)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(ransac_seg_core src/ground_plane.cpp)
target_include_directories(ransac_seg_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(ransac_seg_core PUBLIC Eigen3::Eigen)

# Scores the port against SemanticKITTI labels, printing the same quantities as
# tools/eval_semantickitti.py so the two are comparable rather than both merely
# "fast". Needs no ROS, so it runs on a bare checkout.
add_executable(bench_ground src/bench_ground.cpp)
target_link_libraries(bench_ground ransac_seg_core)

install(TARGETS ransac_seg_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS bench_ground DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_ground_plane test/test_ground_plane.cpp)
  target_link_libraries(test_ground_plane ransac_seg_core)
endif()

ament_package()
